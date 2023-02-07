import numpy as 


def arbitrage(delta_y: float, cfm_A: CFM, cfm_B: CFM, gamma: float, execute_trade: bool = False):
    """
    Returns the profit of the trade, and executes the trade if execute_trade = True
    """
    shadow_price_with_fee = grad_(Gy, argnums=0, gamma=gamma)
    price_A = shadow_price(0., cfm_A.x, cfm_A.y, 'x')#.item()
    #price_A = shadow_price_with_fee(0., cfm_A.x, cfm_A.y, 'x')#.item()
    price_B = shadow_price_with_fee(0., cfm_B.x, cfm_B.y, 'x')#.item()

    if price_A < price_B: # buy x in A and sell it in B
        delta_x_A = Gx(delta_y=delta_y, x=cfm_A.x, y=cfm_A.y, buy='x')
        final_delta_y = Gy(delta_x=1/(2-gamma)*delta_x_A, x=cfm_B.x, y=cfm_B.y, buy='y') # already considering the fees paid in x here
    else: # buy x in B and sell it in A
        delta_x_B = Gx(delta_y=1/(2-gamma)*delta_y, x=cfm_B.x, y=cfm_B.y, buy='x')
        final_delta_y = Gy(delta_x=delta_x_B, x=cfm_A.x, y=cfm_A.y, buy='y') # already considering the fees paid in x here

    if execute_trade:
        if price_A < price_B: # buy x in A and sell it in B
            cfm_A.x = cfm_A.x - delta_x_A
            cfm_A.y = cfm_A.y + delta_y
            #cfm_A.fees.y = cfm_A.fees.y# + (1-gamma) * delta_y

            cfm_B.x = cfm_B.x + 1/(2-gamma)*delta_x_A
            cfm_B.fees.x = cfm_B.fees.x + (1-gamma)/(2-gamma)*delta_x_A
            cfm_B.y = cfm_B.y - final_delta_y
        else: # buy x in B and sell it in A
            cfm_B.x = cfm_B.x - delta_x_B
            cfm_B.y = cfm_B.y + 1/(2-gamma)*delta_y
            cfm_B.fees.y = cfm_B.fees.y + (1-gamma)/(2-gamma) * delta_y

            cfm_A.x = cfm_A.x + delta_x_B
            #cfm_A.fees.x = cfm_A.fees.x + (1-gamma)/(2-gamma)*delta_x_B
            cfm_A.y = cfm_A.y - final_delta_y


    return final_delta_y - delta_y - (1-gamma) * delta_y


def minus_profit(delta_y: float, cfm_A: Pool, cfm_B: Pool, gamma: float):
    """
    Auxiliary function used to maximize profit (i.e. minimize -1 * profit)
    """
    return -1 * arbitrage(delta_y, cfm_A, cfm_B, gamma, False)
