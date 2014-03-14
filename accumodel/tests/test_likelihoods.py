import unittest
import numpy as np
from scipy import stats

from accumodel import likelihoods

def test_invgauss_sampling():
    samples = 5000
    likelihoods.init_rands((samples, 3), seed=101)
    for i in range(10):
        a = np.random.rand()*10.
        v = np.random.rand()*10.
        sigma = np.random.rand()*10.
        mu = a / v
        lam = a**2 / sigma**2

        samples = likelihoods.invgauss(0, a, v, sigma=sigma)
        # needs transform: https://github.com/scipy/scipy/issues/2367#issuecomment-17028905
        D, p_value = stats.kstest(samples, stats.invgauss(mu/lam, loc=0, scale=lam).cdf)

        assert p_value > .05, "Not from the same distribution. Params: a=%f, v=%f, mu=%f, lam=%f" % (a, v, mu, lam)

def test_invgauss_pdf():
    for i in range(10):
        a = np.random.rand()*10.
        v = np.random.rand()*10.
        sigma = np.random.rand()*10.
        mu = a / v
        lam = a**2 / sigma**2
        x = np.linspace(0.1, 100, 1000)
        y1 = np.exp(likelihoods.invgauss_logpdf(x, 0, a, v, sigma=sigma, p_outlier=0.))
        # needs transform: https://github.com/scipy/scipy/issues/2367#issuecomment-17028905
        y2 = stats.invgauss(mu/lam, loc=0, scale=lam).pdf(x)

        np.testing.assert_array_almost_equal(y1, y2)

def test_pda_single():
    pass
