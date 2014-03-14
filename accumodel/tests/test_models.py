import unittest
import numpy as np
from scipy import stats

from accumodel import models
from accumodel import estimators

def test_wald_anti_pda():
    params = {'t': .3,
              'a': 2.,
              'v_pro': 4.,
              'v_anti': 6.,
              't_anti': .3,
    }
    est = estimators.OptimizeEstimator(models.WaldAntiPDA)
    data = est.gen_data(params)
    model = models.WaldAntiPDA(data)
    model.approximate_map()
