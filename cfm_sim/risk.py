import numpy as np

def VaR(alpha: float, x: np.ndarray):
    """
    
    VaR_alpha(X) = -F_X^{-1}(alpha)
    """
    percentile = np.percentile(x, alpha*100)
    return -percentile


def ES(lam: float, x: np.ndarray):
    """
    Expected Shortfall
    """
    empirical_var = VaR(alpha=lam, x=x)
    mask = x[x < -empirical_var]

    N = len(x)

    expected_shortfall = -1/lam * (1/N * mask.sum() + empirical_var*(1/N * len(mask) - lam))
    return expected_shortfall
