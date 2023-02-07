import numpy as np

from lib.cfm import Exchange


class Price:

    def __get__(self, obj, objtype=None):
        S = obj.S_past[-1]
        return S, None, None


class CEX(Exchange):

    price_descr = Price()

    def __init__(self, sigma: float, init_price: float, mu: float = 0, **kwargs):
        super().__init__(x=0,y=0,init_price=init_price) # we don't care about reserves, it is infinitely liquid
        self.dt = 0.01
        self.mu = mu
        self.sigma = sigma

        self.gamma = 1. # no fees in the cex

    def V(self):
        pass

    def fee(self):
        pass

    def get_delta_x(self, delta_y,):
        """
        S is price of x in y
        """
        S = self.S_past[-1]
        return -delta_y/S # it has to have opposite sign because if delta_y comes in, then delta_x needs to come out, and the opposite

    def get_delta_y(self, delta_x,):
        """
        S is price of x in y
        """ 
        S = self.S_past[-1]
        return -delta_x * S

    def update_S(self, h):
        S = self.S_past[-1] * np.exp((self.mu-0.5*self.sigma**2)*h + self.sigma * np.random.randn() * np.sqrt(h))
        self.S_past.append(S)
        return S

    def fee(self, delta_y, delta_x, **kwargs):

        return {'x':0, 'y':0}

    def execute_trade(self, **kwargs):
        pass

