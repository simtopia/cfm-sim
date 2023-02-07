"""
Utils functions to model a Jump process
"""

import numpy as np
from scipy.stats import expon


def sample_exponential(lam: float, ):
    """
    Sample interarrival times
    """
    u = np.random.rand()
    lag = expon.ppf(u, scale=1/lam)
    return lag



