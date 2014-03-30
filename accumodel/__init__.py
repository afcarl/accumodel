from . import models
from . import likelihoods
from . import estimators

ESTIMATOR = None

def set_estimator(estimator):
    ESTIMATOR = estimator

def instantiate_estimator(*args, **kwargs):
    return ESTIMATOR(*args, **kwargs)