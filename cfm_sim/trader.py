from .exchange import Pool, Exchange
from .grad import grad

from scipy.optimize import root_scalar, minimize
import numpy as np


def _request_trade(delta: float, to: Exchange, asset: str = "y", ):
    """
    make a request to get the price of a trade, from the trader point of view
    """
    if asset=="y": # trade in y: the trader goes long if delta>0, or short if delta<0
        delta_y = delta
        delta_x = -to.get_delta_x(-delta_y) # I change the sign because if the trader goes short, then the CFM goes long, and viceversa
    elif asset=="x": # trade in x: the trader foes long if delta>0, or short if delta<0
        delta_x = delta
        delta_y = -to.get_delta_y(-delta_x) # I change the sign because if the trader goes short, then the CFM goes long, and viceversa
    fee = to.fee(delta_y = -delta_y, delta_x = -delta_x, )
    return delta_x, delta_y, fee


def request_trade(delta: float, to: Exchange, asset: str = 'y', including_fee: bool = False):
    """
    from a trader point of view
    """
    if delta < 0 and including_fee: # i.e. the trader goes short, and we want the delta to include the fee
        true_delta = 1/(2-to.gamma)*delta
    else:
        true_delta = delta
    return _request_trade(true_delta, to, asset)
    
        
    



class Trader(Pool):
    """Trader class
    """

    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y) # wallet of the trader
    

    def send_trade(self, delta: float, to: Exchange, asset: str = "y", including_fee: bool = False):
        """Send a trade to the cfm. delta_y can be positive (the trader goes long in y), or negative (the trader goes short in y)
        """
        delta_x, delta_y, fee = request_trade(delta, to, asset, including_fee)
        # update trader's wallet
        self.y = self.y + delta_y - fee['y']
        self.x = self.x + delta_x - fee['x']

        # update CFM's reserves and fees
        to.execute_trade(delta_x = -delta_x, delta_y = -delta_y, fee = fee) # we put minus because if the trader goes short in one asset, then the market goes long in that asset.

        return "success"

    def send_trade_gbm(self, to: Exchange, sigma: float, h: float, with_drift: bool=False, asset: str = "y"):
        delta_x = noise_trade_size_gbm(to, sigma, h, with_drift)
        self.send_trade(delta_x, to, asset='x')

    def send_trade_price_sensitive(self, to_illiquid: Exchange, to_liquid: Exchange, ):
        delta_y = 0.5*np.random.randn()
        delta_x, _, fee = request_trade(delta_y, to_illiquid, asset = 'y', including_fee = True) 
        delta_x_illiquid = np.abs(delta_x) + fee['x']
        
        delta_x, _, fee = request_trade(delta_y, to_liquid, asset='y', including_fee = True)
        delta_x_liquid = np.abs(delta_x) + fee['x']
        
        u = np.random.rand()
        
        if delta_y > 0: # the trader goes long in y
            if u<0.9: # with prob 0.8 the noise trade will be rational
                chosen_exchange = to_illiquid if delta_x_illiquid <= delta_x_liquid else to_liquid
            else:
                chosen_exchange = to_liquid if delta_x_illiquid <= delta_x_liquid else to_illiquid

        else: # the trader foes short in y
            if u<0.9: # with prob 0.8 the noise trader will be rational
                chosen_exchange = to_liquid if delta_x_illiquid <= delta_x_liquid else to_illiquid
            else:
                chosen_exchange = to_illiquid if delta_x_illiquid <= delta_x_liquid else to_liquid

        self.send_trade(delta_y, chosen_exchange, asset='y', including_fee=True)
        return 'success'




def get_profit(delta_y: float, to_liquid: Exchange, to_illiquid: Exchange, minimize: bool = False,):
    """
    The arbitrageur goes short in y (i.e. delta_y < 0), in one of the cfms, and goes long in y back in the other cfm. 
    We need to be careful because the illiquid cfm charges a fee, while the other one does not.
    """
    cost_with_fee = lambda x: -(2-to_illiquid.gamma) * to_illiquid.get_delta_y(x)
    price_illiquid = grad(cost_with_fee, argnums=0)(0)
    #price_liquid = to_liquid.S()
    price_liquid = to_liquid.price_descr
    if price_liquid < price_illiquid:
        delta_x, _, _ = request_trade(delta=delta_y, to=to_liquid, including_fee=True, asset="y")
        trade1 = {'delta':delta_y, 'to':to_liquid, 'including_fee': True, 'asset': 'y'}
        # get the values of the trade in the illiquid CFM (with premium fee)
        _, delta_y_back, fees2 = request_trade(delta = -delta_x, to=to_illiquid, asset="x", including_fee=True,)
        trade2 = {'delta':-delta_x, 'to': to_illiquid, 'asset': 'x', 'including_fee': True}
    else:
        delta_x, _, fees1 = request_trade(delta=delta_y, to=to_illiquid, including_fee=True, asset="y")
        trade1 = {'delta':delta_y, 'to':to_illiquid, 'including_fee': True, 'asset': 'y'}
        # get the values of the trade in the illiquid CFM (with premium fee)
        _, delta_y_back, fees2 = request_trade(delta = -delta_x+fees1['x'], to=to_liquid, asset="x", including_fee=True,)
        trade2 = {'delta':-delta_x, 'to': to_liquid, 'asset': 'x', 'including_fee': True}
    profit = delta_y + delta_y_back - fees2['y']
    if minimize:
        return -profit
    else:
        return profit, trade1, trade2



class Arbitrageur(Trader):

    def __init__(self, x: float, y: float):
        super().__init__(x=x,y=y)

    
    def execute_arbitrage(self, to_liquid: Exchange, to_illiquid: Exchange):
        
        cost_with_fee = lambda x: -(2-to_illiquid.gamma) * to_illiquid.get_delta_y(x)
        price_illiquid = grad(cost_with_fee, argnums=0)(0)
        #price_liquid = to_liquid.S()
        price_liquid = to_liquid.price_descr
        
        try:
            argmin = minimize(fun = get_profit, x0=0, args=(to_liquid, to_illiquid, True))
            profit, trade1, trade2 = get_profit(delta_y=argmin.x.item(), to_liquid=to_liquid, to_illiquid=to_illiquid, minimize=False)
            if profit > 0 and argmin.x.item() < 0:
                self.send_trade(**trade1)
                self.send_trade(**trade2)
        except KeyError:
            pass


def noise_trade_size_gbm(cfm: Exchange, sigma: float, h: float, with_drift: bool = False):
    """
    Cooked up noise trade size such that the cfm shadow price is a GBM woith no drift
    
    Parameters
    ----------
    cfm: Pool
        cfm.x and cfm.y are the reserves in x and y
    gamma: float
        1 - fee
    sigma: float
        volatility of gbm
    
    Returns
    -------
    eta_x: float
        Trade size in x
    """
    rng = np.random.default_rng()
    z = rng.normal(0, 1)
    
    S, dSdx, dSdy = cfm.price_descr

    def taylor_1st(delta_x,):
        
        if with_drift:
            mu = 0.01
            #return (-cfm.dSdx() * delta_x + cfm.dSdy() * cfm.S() * delta_x) - (mu*cfm.S()*h + sigma * np.sqrt(h)*cfm.S()*z)
            return (-dSdx * delta_x + dSdy * S * delta_x) - (mu*S*h + sigma * np.sqrt(h)*S*z)
        else:    
            #return (-cfm.dSdx() * delta_x + cfm.dSdy() * cfm.S() * delta_x) - (sigma * np.sqrt(h)*cfm.S()*z)
            return (-dSdx * delta_x + dSdy * S * delta_x) - (sigma * np.sqrt(h)*S*z)
    
    delta_x = root_scalar(taylor_1st, x0=0,x1=20.)
    return delta_x.root.item()
