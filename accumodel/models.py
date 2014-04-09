import numpy as np
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

from . import likelihoods

class WaldAntiPDA(hddm.models.HLBA):
    slice_widths = {'a':1, 't':0.01, 'a_std': 1, 't_std': 0.15,
                    'v_pro': 1, 'v_anti': 1, 'v_stop': 1,
                    't_anti': .01}

    param_ranges = OrderedDict([('t', (0, .5)),
                                ('a', (.5, 3.5)),
                                ('v_pro', (0, 5.)),
                                ('v_anti', (0, 5.)),
                                ('t_anti', (0, .5)),
                               ])

    def _create_family_beta(self, name, value=.5, g_value=.5, g_mean=.5, g_certainty=2,
                           var_alpha=1, var_beta=1, var_value=.1):
        """Similar to create_family_normal() but beta for the subject
        and group mean nodes. This is useful when the parameter space
        is restricted from [0, 1].

        See create_family_normal() help for more information.

        """

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_mean = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty,
                      value=g_value, depends=self.depends[name])

            g_certainty = Knode(pm.Gamma, '%s_certainty' % name,
                                alpha=var_alpha, beta=var_beta, value=var_value)

            alpha = Knode(pm.Deterministic, '%s_alpha' % name, eval=lambda mean, certainty: mean*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            beta = Knode(pm.Deterministic, '%s_beta' % name, eval=lambda mean, certainty: (1-mean)*certainty,
                      mean=g_mean, certainty=g_certainty, plot=False, trace=False, hidden=True)

            subj = Knode(pm.Beta, '%s_subj'%name, alpha=alpha, beta=beta,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=False)

            knodes['%s'%name]            = g_mean
            knodes['%s_certainty'%name]  = g_certainty
            knodes['%s_alpha'%name]      = alpha
            knodes['%s_beta'%name]       = beta
            knodes['%s_bottom'%name]     = subj

        else:
            g = Knode(pm.Beta, name, alpha=g_mean*g_certainty, beta=(1-g_mean)*g_certainty, value=value,
                      depends=self.depends[name])

            knodes['%s_bottom'%name] = g

        return knodes

    def _create_stochastic_knodes_info(self):
        knodes = OrderedDict()
        knodes.update(self._create_family_gamma_gamma_hnormal('t', g_mean=.3, g_std=1., std_std=.5, std_value=0.2, value=0.1))
        knodes.update(self._create_family_gamma_gamma_hnormal('a', g_mean=2., g_std=.75, std_std=2, std_value=0.1, value=2.))
        if 'v_pro' in self.param_ranges:
            knodes.update(self._create_family_gamma_gamma_hnormal('v_pro', g_mean=.5, g_std=1, std_std=2, std_value=0.1, value=.5))
        if 'v_stop' in self.param_ranges:
            knodes.update(self._create_family_gamma_gamma_hnormal('v_stop', g_mean=1.5, g_std=1, std_std=2, std_value=0.1, value=1.5))
        if 'v_anti' in self.param_ranges:
            knodes.update(self._create_family_gamma_gamma_hnormal('v_anti', g_mean=.5, g_std=1, std_std=2, std_value=0.1, value=.5))
        if 't_anti' in self.param_ranges:
            knodes.update(self._create_family_gamma_gamma_hnormal('t_anti', g_mean=.1, g_std=1, std_std=2, std_value=0.1, value=.1))
        if 'p_stop' in self.param_ranges:
            knodes.update(self._create_family_beta('p_stop', value=.9, g_value=.9, g_mean=.9, g_certainty=2, var_alpha=1, var_beta=1, var_value=.1))
        return knodes

    def create_knodes(self):
        knodes = self._create_stochastic_knodes_info()
        knodes['wfpt_pro'] = self._create_pro_knode(knodes)
        knodes['wfpt_anti'] = self._create_anti_knode(knodes)
        return knodes.values()

    def _create_anti_knode(self, knodes):
        parents = OrderedDict()
        for name, knode in knodes.iteritems():
            if name.endswith('_bottom'):
                parents[name[:-7]] = knode

        anti_like = likelihoods.gen_pda_stochastic(gen_func=self.gen_data_anti)
        return Knode(anti_like, 'anti', observed=True, col_name=['rt', 'cond'], **parents)

    def _create_pro_knode(self, knodes):
        parents = OrderedDict()
        parents['t'] = knodes['t_bottom']
        parents['a'] = knodes['a_bottom']
        parents['v_pro'] = knodes['v_pro_bottom']

        return Knode(likelihoods.wald_pro_like, 'pro', observed=True, col_name=['rt', 'cond'], **parents)

    @property
    def aic(self):
        k = len(self.get_stochastics())
        logp = sum([x.logp for x in self.get_observeds()['node']])
        return 2 * k - 2 * logp

    @property
    def bic(self):
        k = len(self.get_stochastics())
        n = len(self.data)
        logp = sum([x.logp for x in self.get_observeds()['node']])
        return -2 * logp + k * np.log(n)

    @staticmethod
    def gen_data_anti(t=.3, a=2., v_pro=1., v_anti=1., t_anti=1.):
        if t < 0 or a < 0 or v_pro < 0 or v_anti < 0 or t_anti < 0:
            return None

        func = likelihoods.fast_invgauss
        x_pro = copy(func(t, a, v_pro, accum=0))
        x_anti = func(t + t_anti, a, v_anti, accum=1)

        idx = x_pro > x_anti
        x_pro[idx] = -x_anti[idx]
        data = x_pro

        return data


class WaldAntiStop(WaldAntiPDA):
    param_ranges = OrderedDict([('t', (0, .5)),
                                ('a', (.5, 3.5)),
                                ('v_pro', (0, 5.)),
                                ('v_stop', (0, 5.)),
                                ('v_anti', (0, 5.)),
                                ('t_anti', (0, .5)),
                               ])
    @staticmethod
    def gen_data_anti(t=.3, a=2., v_pro=1., v_stop=1., v_anti=1., t_anti=1.):
        if t < 0 or a < 0 or v_pro < 0 or v_anti < 0 or t_anti < 0:
            return None

        func = likelihoods.fast_invgauss
        x_pro = copy(func(t, a, v_pro, accum=0))
        x_anti = func(t + t_anti, a, v_anti, accum=1)
        x_stop = func(t, a, v_stop, accum=2)

        x_pro[x_pro > x_stop] = np.inf

        idx = x_pro > x_anti
        x_pro[idx] = -x_anti[idx]
        data = x_pro

        return data

class WaldAntiStopProb(WaldAntiPDA):
    param_ranges = OrderedDict([('t', (0, .5)),
                                ('a', (.5, 3.5)),
                                ('v_pro', (0, 5.)),
                                ('v_stop', (0, 5.)),
                                ('v_anti', (0, 5.)),
                                ('t_anti', (0, .5)),
                                ('p_stop', (0, 1))
                               ])
    @staticmethod
    def gen_data_anti(t=.3, a=2., v_pro=1., v_stop=1., v_anti=1.,
                      t_anti=1., p_stop=1.):
        from scipy.stats import bernoulli
        if t < 0 or a < 0 or v_pro < 0 or v_anti < 0 or t_anti < 0 or p_stop < 0 or p_stop > 1:
            return None

        func = likelihoods.fast_invgauss
        x_pro = copy(func(t, a, v_pro, accum=0))
        x_anti = func(t + t_anti, a, v_anti, accum=1)
        x_stop = func(t, a, v_stop, accum=2)
        if p_stop < 1:
            stop = bernoulli(p_stop).rvs(x_stop.shape)
            x_stop[np.logical_not(stop)] = np.inf

        x_pro[x_pro > x_stop] = np.inf

        idx = x_pro > x_anti
        x_pro[idx] = -x_anti[idx]
        data = x_pro

        return data

class WaldAntiNoStop(WaldAntiPDA):
    param_ranges = OrderedDict([('t', (0, .5)),
                                ('a', (.5, 3.5)),
                                ('v_pro', (0, 3)),
                                ('v_anti', (0, 3)),
                                ('t_anti', (0, .5)),
                               ])
    @staticmethod
    def gen_data_anti(t=.3, a=2., v_pro=1., v_anti=1., t_anti=1.):
        from scipy.stats import bernoulli
        if t < 0 or a < 0 or v_pro < 0 or v_anti < 0 or t_anti < 0:
            return None

        func = likelihoods.fast_invgauss
        x_pro = copy(func(t, a, v_pro, accum=0))
        x_anti = func(t + t_anti, a, v_anti, accum=1)
        idx = x_pro > x_anti
        x_pro[idx] = -x_anti[idx]
        data = x_pro

        return data

class WaldAntiPFC(WaldAntiPDA):
    param_ranges = OrderedDict([('t', (0, .5)),
                                ('a', (.5, 3.5)),
                                ('v_pro', (0, 5.)),
                                ('v_stop', (0, 5.)),
                                ('v_anti', (0, 3.)),
                               ])

    @staticmethod
    def gen_data_anti(t=.3, a=2., v_pro=1., v_stop=1., v_anti=1.):
        from scipy.stats import bernoulli
        if t < 0 or a < 0 or v_pro < 0 or v_anti < 0:
            return None

        func = likelihoods.fast_invgauss
        x_pro = copy(func(t, a, v_pro, accum=0))
        x_pfc = func(t, a, v_stop, accum=1)
        x_anti = func(t, a, v_anti, accum=2) + x_pfc

        x_pro[x_pro > x_pfc] = np.inf

        idx = x_pro > x_anti
        x_pro[idx] = -x_anti[idx]
        data = x_pro

        return data

def estimate_subj((subj_idx, data), debug=False, use_basin=True, **kwargs):
    from . import get_estimator
    est = get_estimator()
    est.setup_model(data, is_group_model=False, **kwargs)
    recovery = est.estimate(data, use_basin=use_basin,
                            minimizer_kwargs={'maxiter': 100000,
                                              'disp': 1}, fall_to_simplex=False,
                            debug=debug)
    recovery.name = subj_idx
    return recovery

def run_single(data, debug=False, use_basin=True, **kwargs):
    from . import get_estimator
    est = get_estimator()
    est.setup_model(data, is_group_model=False, **kwargs)
    recovery = est.estimate(data,
                            use_basin=use_basin,
                            minimizer_kwargs={'maxiter': 100000, 'disp': 1},
                            fall_to_simplex=False, debug=debug)
    return recovery

def recreate_model(data, params, model, **kwargs):
    m = model(data, **kwargs)
    m.mcmc()
    m.set_values(params)
    return m
