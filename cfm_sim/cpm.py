from .exchange import Pool, Exchange
from .grad import grad


def cpm_get_x(y, k):
    """ F(x,y) = xy = k ----> x = k/y"""
    return k / y


def value_cpm(x: float, y: float, S: float):
    """ Value of the pool
    
    Parameters
    ----------
    S: float
        Price of asset x in y in a reference market

    Returns
    -------
    V: float
        V = 2 * (x * y)^{1/2} * S^{1/2}
    """
    return 2 * (x*y*S)**0.5 
    #return y + x * S

class Price:

    def __get__(self, obj, objtype=None):
        S = obj.y / obj.x
        dSdx = -obj.y / obj.x**2
        dSdy = 1 / obj.x

        return S, dSdx, dSdy

class BondCurve:

    def __get__(self, obj, objtype=None):
        k = obj.y * obj.x
        return k

class CPM(Exchange):
    """Uniswap v3. The fees go in a different pool....
    """
    
    price_descr = Price()
    k = BondCurve()
    

    def __init__(self, x: float, y: float, **kwargs):
        try:
            super().__init__(x=x,y=y, init_price=y/x)
        except ZeroDivisionError:
            super().__init__(x=x,y=y,init_price=0)
        
        #self.k = x*y # trading function
        self.gamma = kwargs.get('gamma')
        self.P = grad(self.get_delta_x, argnums=1)
        self.streaming_premia = 0

        self.fees_x_past = [self.fees.x]
        self.fees_y_past = [self.fees.y]

    def V(self, S):
        return value_cpm(self.x, self.y, S)

    def _premium_fee(self, delta_y: float, S: float, **kwargs):
        """
        the fee is given by 
        f: = - [ V(x+delta_x, y+delta_y, S) - V(x, y, S)]
        """
        new_y = self.y + delta_y
        new_x = cpm_get_x(new_y, self.k)
        fee = - (value_cfm(new_x, new_y, S) - self.V(S))
        return {'x':0, 'y':fee} 

    def _percentage_fee(self, delta_y: float, delta_x: float, **kwargs):
        assert(delta_y * delta_x < 0, "{} and {} need to have different sign".format(delta_y, delta_x))
        if delta_y > 0:
            return {'x':0, 'y':(1-self.gamma)*delta_y}
        else:
            return {'x':(1-self.gamma)*delta_x, 'y':0}

    def update_streaming_premia(self, h: float, sigma: float, S: float):

        self.streaming_premia += 0.5**2 / 2 * sigma**2 * self.V(S) * h # we multipy by h because the straming premia formula is a rate. 

    def get_delta_x(self, delta_y: float,):
        y_new = self.y + delta_y
        x_new = cpm_get_x(y_new, self.k)
        delta_x = x_new - self.x
        return delta_x

    def get_delta_y(self, delta_x: float,):
        x_new = self.x + delta_x
        y_new = cpm_get_x(x_new, self.k)
        delta_y = y_new - self.y
        return delta_y


    def fee(self, delta_y, delta_x,**kwargs):
        if self.gamma is not None:
            return self._percentage_fee(delta_y=delta_y, delta_x=delta_x, **kwargs)
        else:
            raise ValueError("Unkown fee type")

    def update_past(self):
        self.S_past.append(self.y / self.x)
        self.x_past.append(self.x)
        self.y_past.append(self.y)

        self.fees_x_past.append(self.fees.x)
        self.fees_y_past.append(self.fees.y)
    
    def execute_trade(self, delta_x, delta_y, fee):
        self.y += delta_y
        self.x += delta_x
        self.fees.y += fee['y']
        self.fees.x += fee['x']
        #self.update_past()
