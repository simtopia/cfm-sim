import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

from .grad import grad


@dataclass
class Pool:
    x: float
    y: float



class Exchange(Pool):

    def __init__(self, x, y, init_price):
        super().__init__(x,y)
        self.fees = Pool(x=0, y=0)
        self.S_past = [init_price]
        self.dt = 0.01
        self.x_past = [self.x]
        self.y_past = [self.y]


    @abstractmethod
    def V(self, S):
        ...

    @abstractmethod
    def fee(self, **kwargs):
        ...

    @abstractmethod
    def get_delta_x(self, delta_y: float):
        ...

    @abstractmethod
    def get_delta_y(self, delta_x: float):
        ...

    @abstractmethod
    def execute_trade(self, **kwargs):
        pass

    
    def realized_vol(self):
        past = np.array(self.S_past)
        ret = (past[1:] - past[:-1]) / past[:-1]
        realized_vol = ret.std() / np.sqrt(self.dt)
        return realized_vol

    def quadratic_variation(self):
        """quadratic variation assuming GBM
        """
        rv = self.realized_vol()
        return rv ** 2 * self.S_past[-1]**2



    









