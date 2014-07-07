import numpy as np
import inspect
from collections import OrderedDict
import pandas as pd

import hddm
from kabuki.utils import stochastic_from_dist
from kabuki.hierarchical import Knode, Hierarchical
np.set_printoptions(suppress=True)

from sklearn.neighbors.kde import KernelDensity
import pymc as pm

from copy import copy

from scipy import stats
from pymc.distributions import new_dist_class

try:
    from numbapro import jit
except:
    jit = lambda x: x

rands = np.empty((1, 1))
u_rands = np.empty((1, 1))
chi_rands = np.empty((1, 1))

def init_rands(size, seed=31337):
    np.random.seed(seed)
    global rands, u_rands, chi_rands, out
    print "Initializing random numbers as", size
    rands = np.random.randn(*size)
    u_rands = np.random.rand(*size)
    chi_rands = np.random.chisquare(1, size=size)
    out = np.empty(size, dtype=np.double)

samples = 10000
init_rands((samples, 3), seed=101)

@jit
def numba_invgauss(t, a, v, u_rands, chi_rands, out, sigma, accum):
    size = u_rands.shape[0]
    i = 0
    mu = a / v
    lam = a**2 / sigma**2

    for i in range(size):
        y2 = chi_rands[i, accum]
        u = u_rands[i, accum]
        r2 = mu/(2.*lam)*(2.*lam+mu*y2+np.sqrt(4.*lam*mu*y2+mu**2.*y2**2.))
        r1 = mu**2./r2
        if u < mu/(mu+r1):
            out[i, accum] = r1 + t
        else:
            out[i, accum] = r2 + t

    return out[:, accum]

def add_outlier_prob(logp, p_outlier=0.05, w_outlier=.1):
    wp_outlier = p_outlier * w_outlier
    logp = np.log(np.exp(logp) * (1 - p_outlier) + wp_outlier)
    logp[np.isnan(logp)] = np.log(wp_outlier)
    return logp

def fast_invgauss(t, a, v, sigma=1, accum=0):
    global u_rands, chi_rands, out
    return numba_invgauss(t, a, v, u_rands, chi_rands, out, sigma, accum)

def invgauss(t, a, v, sigma=1, accum=0):
    global u_rands, chi_rands
    mu = a / v
    lam = a**2 / sigma**2
    y2 = chi_rands[:, accum]
    u = u_rands[:, accum]
    r2 = mu/(2.*lam)*(2.*lam+mu*y2+np.sqrt(4.*lam*mu*y2+mu**2.*y2**2.))
    r1 = mu**2./r2
    idx = u < mu/(mu+r1)
    out = np.empty_like(y2)
    out[idx] = r1[idx]
    out[np.logical_not(idx)] = r2[np.logical_not(idx)]
    return out + t

def invgauss_logpdf(x, t=0.3, a=2., v_pro=1., sigma=1, p_outlier=0.05):
    v = v_pro
    if a < 0 or t < 0 or v < 0 or sigma < 0:
        return -np.inf
    #x = x[x > 0] # censor LB responses
    xt = x - t
    mu = a / v
    lam = a**2 / sigma**2
    logp = .5 * np.log(lam / (2*np.pi*xt**3)) - (lam * (xt - mu)**2) / (2*mu**2*xt)
    return add_outlier_prob(logp, p_outlier=p_outlier)

####################################################

def pda_single(synth_data, data, bandwidth=.1):
    #synth_data = np.log(np.abs(synth_data))[:, np.newaxis]
    #data_log = np.log(np.abs(data))[:, np.newaxis]
    synth_data = synth_data[:, np.newaxis]
    data = data[:, np.newaxis]
    if bandwidth == 'silverman':
        lower, upper = scoreatpercentile(synth_data, [25, 75])
        iqr = upper - lower
        sd = np.std(synth_data)
        bandwidth = .9 * min(sd, iqr/1.34) * len(data)**(-1./5)

    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(synth_data)
    return kde.score_samples(data)

def pda_like_auto(data, func, **kwargs):
    #data = data
    logp = np.empty_like(data)
    #print data
    synth_data = func(**kwargs)
    if synth_data is None:
        return -np.inf

    p = np.mean(synth_data > 0)
    if p > .97 or p < 0.03:
        return -np.inf

    if np.any(data > 0):
        logp[data > 0] = pda_single(synth_data[synth_data > 0], data[data > 0]) + np.log(p)
    if np.any(data < 0):
        logp[data < 0] = pda_single(-synth_data[synth_data < 0], -data[data < 0]) + np.log(1 - p)

    return add_outlier_prob(logp)


def stochastic_from_dist_args(name, logp, args, random=None, logp_partial_gradients={}, dtype=np.dtype('O'), mv=True):
    parent_names = args

    docstr = ""
    distribution_arguments = logp.__dict__

    return new_dist_class(dtype, name, parent_names, {}, docstr, logp,
                          random, mv, {})

def gen_pda_stochastic(gen_func, pda_func=pda_like_auto, selector=None):
    if selector is None:
        selector = lambda value: value.ix[value['cond'] == 'incong', 'rt'].values

    args = inspect.getargspec(gen_func)[0]
    pda = stochastic_from_dist_args(name="Wiener anti pda",
                                    logp=lambda value, **kwargs:
                                    np.sum(pda_func(selector(value), gen_func, **kwargs)),
                                    args=args
    )

    def pdf(self, value):
        params = {p: self.parents[p].value for p in self.parents}
        return np.exp(pda_func(value, gen_func, **params))

    pda.pdf = pdf

    return pda

def gen_simple(t, a, v1, v2):
    if (a < 0) or (v1 < 0) or (v2 < 0):
        return None

    x1 = fast_invgauss(t, a, v1, accum=0)
    x2 = fast_invgauss(t, a, v2, accum=1)
    idx = x1 > x2
    x1[idx] = -x2[idx]
    data = x1

    return data


wald_pro_like = stochastic_from_dist_args(name="Wiener pro pda",
                                          logp=lambda value, **kwargs: \
                                          np.sum(invgauss_logpdf(value.ix[(value['cond'] == 'cong') & (value['rt'] > 0), 'rt'].values, **kwargs)),
                                          args=['t', 'a', 'v_pro']
)

wald_pro_like.pdf = lambda self, value: np.exp(invgauss_logpdf(value,
                                                               **self.parents))
