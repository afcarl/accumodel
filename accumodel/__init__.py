from . import models
from . import likelihoods
from . import estimators

ESTIMATOR = None

def set_estimator(estimator):
    global ESTIMATOR
    ESTIMATOR = estimator

def get_estimator():
    global ESTIMATOR
    return ESTIMATOR