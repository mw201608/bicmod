import numpy as np
import copy
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal, nbinom, poisson, spearmanr
import statsmodels.api as sm
from warnings import catch_warnings

def invlogit(x):
    y = np.exp(x)
    return y / (1 + y)

def invlog(x):
    return np.exp(x)

def zip_cdf(k, mu, psi, loc = 0):
    p1 = poisson.cdf(k = k, mu = mu, loc = loc)
    p1 = psi + p1 * (1 - psi)
    p1[k < 0] = 0
    return p1
def zinb_cdf(k, n, p, psi, loc = 0):
    p1 = nbinom.cdf(k = k, n = n, p = p, loc = loc)
    p1 = psi + p1 * (1 - psi)
    p1[k < 0] = 0
    return p1

#parms: fixed effects in X1, fixed effects in Z1, dispersion in y[, 1], fiexed effects in X2, fixed effects in Z2, dispersion in y[, 2], dependency.
def cop_lik(parms, X1, X2, y, Z1 = None, Z2 = None, offset_x = None, offset_z = None, cop = "gaus", margins = ['nbinom', 'nbinom'], weights = 1, pmf_min = 1e-07, frech_min = 1e-07):
    nx = [X1.shape[1], X2.shape[1]]
    iparm = 0
    lam = np.ndarray(shape=(y.shape[0], 2), dtype = float)
    disp = [0, 0]
    psi = np.ndarray(shape=(y.shape[0], 2), dtype = float)
    for i in range(2):
        par = [parms[j + iparm] for j in range(nx[i])]
        if i == 0:
            X = X1
            Z = Z1
        elif i == 1:
            X = X2
            Z = Z2
        lam[:, i] = invlog(X @ par + offset_x[:, i])
        iparm = iparm + nx[i]
        if margins[i] in ['zip', 'zinb']:
            par = [parms[j + iparm] for j in range(Z.shape[1])]
            psi[:, i] = invlogit(Z @ par + offset_z[:, i])
            iparm = iparm + Z.shape[1]
        if margins[i] in ['nbinom', 'zinb']:
            disp[i] = np.exp(parms[iparm])
            iparm = iparm + 1
    dep = parms[-1]
    if cop == "gaus":
        dep = np.tanh(dep)    
    margin_args = []
    for i in range(2):
        if margins[i] == 'poisson':
            margin_args.append({'q': y[:, i].copy(), 'lambda': lam[:, i].copy()})
        elif margins[i] == 'nbinom':
            margin_args.append({'q': y[:, i].copy(), 'mu': lam[:, i].copy(), 'size': disp[i]})
        elif margins[i] == 'zip':
            margin_args.append({'q': y[:, i].copy(), 'lambda': lam[:, i].copy(), 'psi': psi[:, i].copy()})
        elif margins[i] == 'zinb':
            margin_args.append({'q': y[:, i].copy(), 'mu': lam[:, i].copy(), 'size': disp[i], 'psi': psi[:, i].copy()})
    
    d = cop_pmf(margin_args = margin_args, dep = dep, margins = margins, cop = cop, frech_min = frech_min)
    d = np.clip(d * weights, a_min=pmf_min, a_max=None)
    p0 = -np.sum(np.log(d))
    return p0

def cop_pmf(margin_args, dep = None, margins = ["poisson", "poisson"], cop = "frank", frech_min = 1e-07):
    new_args = []
    for i in range(4):
        new_args.append(copy.deepcopy(margin_args[0]))
        new_args.append(copy.deepcopy(margin_args[1]))
    for i in [2, 5, 6, 7]:
        new_args[i]["q"] = new_args[i]["q"] - 1
    
    cdf_args = []
    iter = 0
    for i in range(4):
        cdf_args.append(new_args[iter:iter + 2])
        iter += 2
    
    p = [cop_cdf(margin_args=w, dep=dep, margins=margins, cop=cop, frech_min=frech_min) for w in cdf_args]
    d = p[0] - p[1] - p[2] + p[3]
    return d


def cop_cdf(margin_args, dep=None, margins=["poisson", "poisson"], cop="frank", frech_min=1e-07):
    u = np.ndarray(shape=(len(margin_args[0]['q']), 2), dtype = float)
    for i in range(2):
        if margins[i] == 'nbinom':
            u[:, i] = nbinom.cdf(margin_args[i]['q'], n = margin_args[i]['size'], p = margin_args[i]['size']/(margin_args[i]['size'] + margin_args[i]['mu']))
        elif margins[i] == 'poisson':
            u[:, i] = poisson.cdf(margin_args[i]['q'], mu = margin_args[i]['lambda'])
        elif margins[i] == 'zinb':
            u[:, i] = zinb_cdf(margin_args[i]['q'], n = margin_args[i]['size'], p = margin_args[i]['size']/(margin_args[i]['size'] + margin_args[i]['mu']), psi = margin_args[i]['psi'])
        elif margins[i] == 'zip':
            u[:, i] = zip_cdf(margin_args[i]['q'], mu = margin_args[i]['lambda'], psi = margin_args[i]['psi'])
    if cop == "frank":
        f = np.exp(- dep * u) - 1
        den = (np.exp(-dep) - 1)
        p = -1 / dep * np.log(1 + f[:, 0] * f[:, 1] / den)
    elif cop == "gaus":
        for i in range(2):
            u[:, i] = np.clip(u[:, i], 1e-16, 1 - 1e-16)
        qu = norm.ppf(u)
        p = multivariate_normal.cdf(x = qu, cov = [[1, dep], [dep, 1]])  # Assuming dep is the correlation coefficient
    p = frech_bounds(p = p, u1 = u[:, 0], u2 = u[:, 1], frech_min = frech_min)
    return p

def frech_bounds(p, u1, u2, frech_min=1e-07):
    fmax = 1 - frech_min
    low = u1 + u2 - 1.0
    low = np.maximum(low, frech_min)
    if frech_min > 0:
        up = np.minimum(np.minimum(u1, u2), fmax)
    else:
        up = np.minimum(u1, u2)
    out = np.minimum(np.maximum(p, low), up)
    return out
def bicmod(X1, X2, y, Z1 = None, Z2 = None, offset_x = None, offset_z = None, cop = "gaus", margins = ['nbinom', 'nbinom'], starts = None, method = None, options = None):
    if offset_x is None:
        offset_x = np.zeros((y.shape[0], 2))
    if offset_z is None:
        offset_z = np.zeros((y.shape[0], 2))
    #
    X1 = np.append(np.ones((X1.shape[0], 1)), X1, axis=1)
    X2 = np.append(np.ones((X2.shape[0], 1)), X2, axis=1)
    if margins[0] not in ['zip', 'zinb']:
        Z1 = None
    else:
        if Z1 is None:
            Z1 = np.ones((y.shape[0], 1))
        else:
            Z1 = np.append(np.ones((y.shape[0], 1)), Z1, axis=1)
    if margins[1] not in ['zip', 'zinb']:
        Z2 = None
    else:
        if Z2 is None:
            Z2 = np.ones((y.shape[0], 1))
        else:
            Z2 = np.append(np.ones((y.shape[0], 1)), Z2, axis=1)
    parm0 = []
    if starts is None:
        for i in range(2):
            if i == 0:
                X = X1
                Z = Z1
            elif i == 1:
                X = X2
                Z = Z2
            nz = 0
            if margins[i] == 'nbinom':
                mod1 = sm.NegativeBinomialP(endog = y[:, i], exog = X, offset = offset_x[:, i])
            elif margins[i] == 'poisson':
                mod1 = sm.Poisson(endog = y[:, i], exog = X, offset = offset_x[:, i])
            elif margins[i] == 'zinb':
                mod1 = sm.ZeroInflatedNegativeBinomialP(endog = y[:, i], exog = X, exog_infl = Z, offset = offset_x[:, i])
                nz = Z.shape[1]
            elif margins[i] == 'zip':
                mod1 = sm.ZeroInflatedPoisson(endog = y[:, i], exog = X, exog_infl = Z, offset = offset_x[:, i])
                nz = Z.shape[1]
            with catch_warnings(record=True) as w:
                res1 = mod1.fit(disp = False)
                if w:
                    print("Warnings from fitting marginal distribution no. " + str(i + 1) + ': ' + str(w[-1].message))
            parm0.extend(res1.params[nz:(nz + X.shape[1])])
            if margins[i] in ['zip', 'zinb']:
                parm0.extend(res1.params[0:Z.shape[1]])
            if margins[i] in ['nbinom', 'zinb']:
                parm0.append(1/res1.params[-1])
        dep = spearmanr(y).statistic
        parm0.append(dep)
    else:
        for i in range(2):
            parm0.extend(starts[i]['coef_X'])
            if margins[i] in ['zip', 'zinb']:
                parm0.extend(starts[i]['coef_Z'])
            if margins[i] in ['nbinom', 'zinb']:
                parm0.append(starts[i]['disp'])
        parm0.append(starts[2]['dep'])
    #print("Parameter initialized:", parm0)
    #
    out = minimize(cop_lik, x0 = parm0, args = (X1, X2, y, Z1, Z2, offset_x, offset_z, cop, margins), method = method, options = options)
    if out.success is False:
        gtol = 1.e-4
        if out.message == 'Desired error not necessarily achieved due to precision loss.':
            if options is None:
                options = {'gtol': gtol}
            else:
                if 'gtol' in options:
                    if options['gtol'] >= gtol:
                        raise Exception("minimize failed with message " + out.message)
                    else:
                        options = {'gtol': gtol}
            print("Retry minimize with gtol " + str(options['gtol']))
            out = minimize(cop_lik, x0 = parm0, args = (X1, X2, y, Z1, Z2, offset_x, offset_z, cop, margins), method = method, options = options)
    if out.success is False:
        raise Exception("minimize failed with message " + out.message)
    dep = out.x[-1]
    if cop == "gaus":
        dep = np.tanh(dep)
    result = []
    iparm = 0
    for i in range(2):
        b1 = {}
        if i == 0:
            X = X1
            Z = Z1
        elif i == 1:
            X = X2
            Z = Z2
        b1['coef.X'] = out.x[[iparm + j for j in range(X.shape[1])]]
        iparm = iparm + X.shape[1]
        if Z is not None:
            b1['coef.Z'] = out.x[[iparm + j for j in range(Z.shape[1])]]
            iparm = iparm + Z.shape[1]
        if margins[i] in ['nbinom', 'zinb']:
            b1['disp'] = np.exp(out.x[iparm])
            iparm = iparm + 1
        result.append(b1.copy())
    result.append({'dep' : dep})
    return Bivfit(result)
class Bivfit:
    def __init__(self, object):
        self.Y1 = {"coef_X" : object[0]['coef.X'], 'coef_Z' : None, 'disp' : None}
        if 'coef.Z' in object[0]:
            self.Y1['coef_Z'] = object[0]['coef.Z']
        if 'disp' in object[0]:
            self.Y1['disp'] = object[0]['disp']
        self.Y2 = {"coef_X" : object[1]['coef.X'], 'coef_Z' : None, 'disp' : None}
        if 'coef.Z' in object[1]:
            self.Y2['coef_Z'] = object[1]['coef.Z']
        if 'disp' in object[1]:
            self.Y2['disp'] = object[1]['disp']
        self.dep = object[2]['dep']
    def summary(self):
        print("Count Y1")
        print("  X coefficients:", self.Y1['coef_X'])
        if self.Y1['coef_Z'] is not None:
            print("  Zero infl coefficients:", self.Y1['coef_Z'])
        if self.Y1['disp'] is not None:
            print("  Dispersion:", self.Y1['disp'])
        print("Count Y2")
        print("  X coefficients:", self.Y2['coef_X'])
        if self.Y2['coef_Z'] is not None:
            print("  Zero infl coefficients:", self.Y2['coef_Z'])
        if self.Y2['disp'] is not None:
            print("  Dispersion:", self.Y2['disp'])
        print("Dependence:", self.dep)
