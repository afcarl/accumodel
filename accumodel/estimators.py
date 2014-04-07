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

from . import models
from . import likelihoods

class AbstractEstimator(object):
    def __init__(self, model_class):
        self.model_class = model_class

    def setup_model(self, data, **kwargs):
        self.model = self.model_class(data, **kwargs)
        self.model.mcmc()

    def _raw_anti_gen(self, params):
        return self.model_class.gen_data_anti(**params)

    def _raw_pro_gen(self, params):
        return likelihoods.invgauss(params['t'], params['a'],
                                    params['v_pro'])

    def gen_data(self, params=None, seed=123, size=500):
        import numpy as np
        import pandas as pd
        likelihoods.init_rands((likelihoods.samples, 3), seed=seed)

        raw_data_pro = self._raw_pro_gen(params)
        raw_data_anti = self._raw_anti_gen(params)

        anti_trials = raw_data_anti[np.abs(raw_data_anti) < 15.][:size]
        data_anti = pd.DataFrame({'rt': anti_trials, 'response': anti_trials > 0, 'cond': ['incong'] * size})
        pro_trials = raw_data_pro[np.abs(raw_data_pro) < 15.][:size]
        data_pro = pd.DataFrame({'rt': pro_trials, 'response': pro_trials > 0, 'cond': ['cong'] * size})
        data = pd.concat([data_pro, data_anti], axis=0)

        self.setup_model(data)

        return data

    def estimate(self, *args, **kwargs):
        raise NotImplementedError('estimate has to be overwritten.')


class OptimizeEstimator(AbstractEstimator):
    def estimate(self, data, minimizer='Powell', use_basin=False,
                 minimizer_kwargs=None, basin_kwargs=None, init=None, **kwargs):
        #assert self.model.nodes_db.ix['anti', 'node'].parents['v_pro'] is self.model.nodes_db.ix['pro', 'node'].parents['v']
        if init is not None:
            self.model.set_values(init)
        print self.model.mc.logp

        ## basin_hopping
        def in_param_range(p, param_ranges=self.model.param_ranges):
            return np.all([(pi >= l) and (pi <= u) for pi, (l, u) in zip(p, param_ranges.values())])

        def step(x, param_ranges=self.model.param_ranges, model=self.model, T=.1):
            print 'x =', x
            from copy import copy

            orig = copy(model.values)
            param_names = model.get_stochastics()['knode_name']
            print param_names

            scaling = np.array([param_ranges[name][1] - param_ranges[name][0] for name in param_names])
            print param_ranges
            def set_values(a):
                for val, (name, stoch) in zip(a, model.iter_stochastics()):
                    stoch['node'].set_value(val)

            set_values(x)
            old_logp = model.logp

            tries = 0
            while tries < 100:
                #vals = model.draw_from_prior()
                inner_tries = 0
                while inner_tries < 50:
                    vals = (np.random.randn(x.shape[0]) * scaling * T) + x
                    if in_param_range(vals):
                        break
                    inner_tries += 1

                print 'prop =', vals
                set_values(vals)
                try:
                    proposed_logp = model.logp
                    break
                except pm.ZeroProbability:
                    continue
                tries += 1

            model.set_values(orig)
            return vals

        def print_fun(x, f, accepted):
            print("at minima %.4f accepted %d" % (f, int(accepted)))
            print("minima =", x)

        if basin_kwargs is None:
            basin_kwargs = {'take_step': step,
                            'T': 2,
                            'niter': 5,
                            'callback': print_fun,
                           }

        self.model.approximate_map(use_basin=use_basin, minimizer=minimizer,
                                   basin_kwargs=basin_kwargs, minimizer_kwargs=minimizer_kwargs, **kwargs)


        values = self.model.values
        for k, v in values.iteritems():
            print v
            if np.abs(v) < 1e-3:
                values[k] = 1e-4
        return pd.Series(values.values(), index=values.keys())
