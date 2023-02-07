"""
gradient functions necessary to calculate infinitesimal exchange rates for example in Uniswap.
I use finite differences as it is enough (and fast) for our simulations... Inspired in grad() from JAX
"""

import numpy as np


def grad(F, argnums: int=0, ):
    """ Gradient of F(*args) in terms of args[argnums]
    We just use finite differences
    """
    def my_grad(*args, **kwargs):
        h = 0.001
        my_args = list(args).copy()
        my_args[argnums] += h
        return (F(*my_args) - F(*args)) / h
    return my_grad
