import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .exchange import Pool, Exchange
from .cpm import CPM
from .grad import grad



class Price:

    def __get__(self, obj, objtype=None):
        
        S = (obj.y + obj.k * np.sqrt(obj.P_l)) / (obj.x + obj.k / np.sqrt(obj.P_u))
        return S
        

class Constant:
    """
    (x + k/sqrt(P_u))(k + k * sqrt(P_l)) = k^2 --> k^2(1 - sqrt(P_l) / sqrt(P_u)) + k(-x*sqrt(P_l) - y/sqrt(P_u)) - x*y = 0
    A = (1 - sqrt(P_l) / sqrt(P_u))
    B = (-x*sqrt(P_l) - y/sqrt(P_u))
    C = x*y

    Then k=(-B + sqrt(B^2 - 4AC)) / 2A
    """

    def __get__(self, obj, objtype=None):

        a = 1 - np.sqrt(obj.P_l) / np.sqrt(obj.P_u)
        b = -obj.x * np.sqrt(obj.P_l) - obj.y / np.sqrt(obj.P_u) 
        c = -obj.x * obj.y

        k = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        return k

class MaxReserves:

    def __get__(self, obj, objtype=None):
        k = obj.k
        x_max = k / np.sqrt(obj.P_l) - k / np.sqrt(obj.P_u)
        y_max = k * np.sqrt(obj.P_u) - k * np.sqrt(obj.P_l)

        return x_max, y_max


class CPM_CL(Exchange):
    """CPM with concentrated liquidity, i.e. a CPM with the reserves shifted so that it only covers the prices where it is defined. 
    """
    
    price_descr = Price()
    k = Constant()
    max_reserves = MaxReserves()
    

    def __init__(self, x: float, y: float, P_l: float, P_u: float, **kwargs):
        #try:
        #    super().__init__(x=x,y=y, init_price=y/x)
        #except ZeroDivisionError:
        super().__init__(x=x,y=y,init_price=0)
        
        # Lower and upper bound of price range
        self.P_l, self.P_u = P_l, P_u

        self.gamma = kwargs.get('gamma')
        self.streaming_premia = 0

        self.fees_x_past = [self.fees.x]
        self.fees_y_past = [self.fees.y]

    def V(self, S):
        
        k = self.k
        V = k * (np.sqrt(S) - np.sqrt(self.P_l)) + k * S * (1 / np.sqrt(S) - 1 / np.sqrt(self.P_u)) 
        return V

    def update_streaming_premia(self, h: float, sigma: float, S: float):

        self.streaming_premia += 0.5**2 / 2 * sigma**2 * self.V(S) * h # we multipy by h because the straming premia formula is a rate. 

    def get_delta_x(self, delta_y: float,):
        k = self.k
        y_new = self.y + delta_y
        x_new = k**2 / (y_new + k * np.sqrt(self.P_l)) - k / np.sqrt(self.P_u)
        delta_x = x_new - self.x
        return delta_x

    def get_delta_y(self, delta_x: float,):
        k = self.k
        x_new = self.x + delta_x
        y_new = k**2 / (x_new + k / np.sqrt(self.P_u)) - k * np.sqrt(self.P_l)
        delta_y = y_new - self.y
        return delta_y
    
    def _percentage_fee(self, delta_y: float, delta_x: float, **kwargs):
        #assert delta_y * delta_x < 0, "{} and {} need to have different sign".format(delta_y, delta_x)
        if delta_y > 0:
            return {'x':0, 'y':(1-self.gamma)*delta_y}
        else:
            return {'x':(1-self.gamma)*delta_x, 'y':0}


    def update_past(self):
        self.S_past.append(self.price_descr)
        self.x_past.append(self.x)
        self.y_past.append(self.y)

        self.fees_x_past.append(self.fees.x)
        self.fees_y_past.append(self.fees.y)
    


class UniswapV3():

    def __init__(self, range_prices: np.ndarray, init_price: float, gamma: float, x_virt: float,):
        """
        We initialise all the ticks. 
        x is risky asset, y is numeraire
        """
        self.range_prices = range_prices
        self.cpm_cl = {}
        for i, p_l in enumerate(range_prices[:-1]):
            p_u = range_prices[i+1]
            if init_price >= p_l and init_price < p_u:
                #x_virt = 10
                y_virt = x_virt * init_price
                x = x_virt - np.sqrt(x_virt * y_virt) / np.sqrt(p_u)
                y = y_virt - np.sqrt(x_virt * y_virt) * np.sqrt(p_l)
                self.cpm_cl[p_l] = CPM_CL(x=x, y=y, P_l=p_l, P_u=p_u, gamma=gamma)
            elif p_l < init_price:
                self.cpm_cl[p_l] = CPM_CL(x=0, y=p_u*2, P_l=p_l, P_u=p_u, gamma=gamma)
            elif p_u >= init_price:
                self.cpm_cl[p_l] = CPM_CL(x=2, y=0, P_l=p_l, P_u=p_u, gamma=gamma)

        self.fees = Pool(0,0)
        self.gamma = gamma
        

    def get_delta_x(self, delta_y: float):
        """
        We have to split the trades
        """
        
        for k, p_l in enumerate(self.range_prices[:-1]):
            if self.cpm_cl[p_l].x > 0 and self.cpm_cl[p_l].y > 0:
                break
        delta_x = 0
        while True:
            
            p_l = self.range_prices[k]
            x_max, y_max = self.cpm_cl[p_l].max_reserves
            
            if self.cpm_cl[p_l].y + delta_y < 0:
                delta_x += (x_max - self.cpm_cl[p_l].x) 
                k = k - 1
                delta_y = self.cpm_cl[p_l].y + delta_y
            elif self.cpm_cl[p_l].y + delta_y > y_max:
                delta_x += - self.cpm_cl[p_l].x 
                k = k + 1
                delta_y = self.cpm_cl[p_l].y + delta_y - y_max
            else:
                delta_x += self.cpm_cl[p_l].get_delta_x(delta_y)
                break
        return delta_x

    def get_delta_y(self, delta_x: float, execute=False):
        """We have to split the trades
        """

        for k, p_l in enumerate(self.range_prices):
            if self.cpm_cl[p_l].x > 0 and self.cpm_cl[p_l].y > 0:
                break
        
        delta_y = 0

        while True:

            p_l = self.range_prices[k]
            x_max, y_max = self.cpm_cl[p_l].max_reserves

            if self.cpm_cl[p_l].x + delta_x < 0:
                delta_y += (y_max - self.cpm_cl[p_l].y) 
                k = k + 1
                delta_x = self.cpm_cl[p_l].x + delta_x

                if execute:
                    self.cpm_cl[p_l].fees.y += (y_max - self.cpm_cl[p_l].y) * (1-self.cpm_cl[p_l].gamma)
                    self.cpm_cl[p_l].x = 0
                    self.cpm_cl[p_l].y = y_max

            elif self.cpm_cl[p_l].x + delta_x > x_max:
                delta_y += -self.cpm_cl[p_l].y 
                k = k-1
                delta_x = self.cpm_cl[p_l].x + delta_x - x_max

                if execute:
                    self.cpm_cl[p_l].fees.x += delta_x * (1-self.cpm_cl[p_l].gamma)
                    self.cpm_cl[p_l].x = x_max
                    self.cpm_cl[p_l].y = 0

            else:
                delta_y += self.cpm_cl[p_l].get_delta_y(delta_x)
                if execute:
                    new_delta_y = self.cpm_cl[p_l].get_delta_y(delta_x)
                    self.cpm_cl[p_l].y += new_delta_y
                    self.cpm_cl[p_l].x += delta_x
                    if delta_x >= 0:
                        self.cpm_cl[p_l].fees.x += delta_x * (1-self.cpm_cl[p_l].gamma)
                    else:
                        self.cpm_cl[p_l].fees.y += new_delta_y * (1-self.cpm_cl[p_l].gamma)
                break
        return delta_y

    def execute_trade(self, delta_x, **kwargs):
        return self.get_delta_y(delta_x = delta_x, execute = True)
    
    def _percentage_fee(self, delta_y: float, delta_x: float, **kwargs):
        #assert delta_y * delta_x < 0, "{} and {} need to have different sign".format(delta_y, delta_x)
        if delta_y > 0:
            return {'x':0, 'y':(1-self.gamma)*delta_y}
        else:
            return {'x':(1-self.gamma)*delta_x, 'y':0}

    def fee(self, delta_y, delta_x,**kwargs):
        if self.gamma is not None:
            return self._percentage_fee(delta_y=delta_y, delta_x=delta_x, **kwargs)
        else:
            raise ValueError("Unkown fee type")

    def plot_reserves(self):
        price_range, cpm_cl_ = zip(*self.cpm_cl.items())
        x = [c.x for c in cpm_cl_]
        y = [c.y for c in cpm_cl_]
        df = pd.DataFrame(dict(price_range=price_range, x=x,y=y))
        fig, ax = plt.subplots()
        df.plot(x='price_range', y=['x','y'], kind='bar', ax=ax)
        ax.set_xlabel(r'Lower bound of price ticks')
        return fig, ax

    def get_price(self):
        
        for tick, cpm in self.cpm_cl.items():
            if cpm.x > 0 and cpm.y > 0:
                return cpm.price_descr

    def print(self):
        for tick, cpm in self.cpm_cl.items():
            print('tick = {}, x = {}, y = {}'.format(tick, cpm.x, cpm.y))





class Position():

    """
    General position on Uniswap v3: 
        - either remove or add liquidity at a certain tick range
    """

    def __init__(self, cpm: UniswapV3, tick: float, beta: float):
        """
        Parameters
        ----------
        beta: k / k_{t-} where k is the constant of cpm.cpm_cl[tick]
            in other works, new liquidty oover old liquidity
        """
        self.beta = beta
        self.x = (1-beta) * cpm.cpm_cl[tick].x
        self.y = (1-beta) * cpm.cpm_cl[tick].y

        cpm.cpm_cl[tick].x *= beta
        cpm.cpm_cl[tick].y *= beta

        self.cpm = cpm
        self.tick = tick
        self.tick_pu = min(cpm.range_prices[cpm.range_prices > tick])

        self.streaming_premia = 0


    def payoff_sim(self, S):
        """
        Exact payoff for different values of S to avoid doing a simulation
        """
        if S<self.tick:
            y=0
            x = self.cpm.cpm_cl[self.tick].k**2 / (self.cpm.cpm_cl[self.tick].k * np.sqrt(self.tick)) - self.cpm.cpm_cl[self.tick].k / np.sqrt(self.tick_pu)
        elif S>=self.tick_pu:
            y = self.cpm.cpm_cl[self.tick].k**2 / (self.cpm.cpm_cl[self.tick].k / np.sqrt(self.tick_pu)) - self.cpm.cpm_cl[self.tick].k * np.sqrt(self.tick)
            x=0
        else:
            x = self.cpm.cpm_cl[self.tick].k / np.sqrt(S) - self.cpm.cpm_cl[self.tick].k / np.sqrt(self.tick_pu)
            y = self.cpm.cpm_cl[self.tick].k**2 / (x + self.cpm.cpm_cl[self.tick].k / np.sqrt(self.tick_pu)) - self.cpm.cpm_cl[self.tick].k * np.sqrt(self.tick)

        closeup_x = 1 / self.beta * x - x
        closeup_y = 1 / self.beta * y - y

        payoff = (self.x - closeup_x) * S + (self.y - closeup_y) 

        return payoff

    def payoff(self, S):
        closeup_x = 1 / self.beta * self.cpm.cpm_cl[self.tick].x - self.cpm.cpm_cl[self.tick].x
        closeup_y = 1 / self.beta * self.cpm.cpm_cl[self.tick].y - self.cpm.cpm_cl[self.tick].y

        payoff = (self.x - closeup_x) * S + (self.y - closeup_y)
        return payoff

    def close(self, S):
        
        closeup_x = 1 / self.beta * self.cpm.cpm_cl[self.tick].x - self.cpm.cpm_cl[self.tick].x
        closeup_y = 1 / self.beta * self.cpm.cpm_cl[self.tick].y - self.cpm.cpm_cl[self.tick].y

        #payoff = (self.x - closeup_x) * S + (self.y - closeup_y)

        self.cpm.cpm_cl[self.tick].x += closeup_x
        self.cpm.cpm_cl[self.tick].y += closeup_y

        self.x -= closeup_x
        self.y -= closeup_y

        payoff = self.x * S + self.y 
        
        return payoff

    def update_streaming_premia(self, h: float, sigma: float, S: float):
        payoff = lambda x: self.payoff_sim(x)
        curv = grad(grad(payoff))(S)
        self.streaming_premia += 0.5 * sigma * S**2 * curv * h
        return 0



class ITM_call(Position):


    def __init__(self, cpm: UniswapV3, tick: float, beta: float,):
        
        assert beta < 1, "beta should be negative, position holder is taking money out of Uniswap"
        super().__init__(cpm, tick, beta)
        self.long_x()


    def long_x(self,):

        x = self.cpm.cpm_cl[self.tick].k**2 / (self.cpm.cpm_cl[self.tick].k * np.sqrt(self.tick)) - self.cpm.cpm_cl[self.tick].k / np.sqrt(self.tick_pu)
        delta_x = 1/self.beta * x - x
        delta_y = -self.cpm.execute_trade(-delta_x)
        self.x += delta_x
        self.y += delta_y
        return 0


