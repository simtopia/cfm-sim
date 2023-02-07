"""
Simulation with two CFMs (one very liquid, and one illiquid), and one arbitrageur

At time 0: the two CFMs have the same price
In each time step:
    - Noise trader makes a (cooked) trade in the liquid exchange so that the price follows a GBM with zero drift and some vol
    - Arb comes and brings the prices together (modulo fees...)

"""




import numpy as np
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd

from cfm_sim.trader import Trader, Arbitrageur
from cfm_sim.cpm import CPM
from cfm_sim.cex import CEX
from cfm_sim.poisson_process import sample_exponential



def run_sim(init_price: float, N_mc: int, gammas, sigmas, lambdas, ts: np.ndarray = np.linspace(0,1,100)):

    sim = defaultdict(list)
    uniswap_fees = defaultdict(list)

    # SIMULATE N_MC scenarios for each gamma, sigma
    for gamma, sigma, lam in product(gammas, sigmas, lambdas): #[0.97]: #[ 0.9,0.95,0.96,0.97,0.98,0.99,1.]:
        print('gamma = {}, sigma={}'.format(gamma, sigma))
        for n_mc in tqdm(range(N_mc)):
            # init CFM pool
            cpm_illiquid = CPM(x=10., y=10. * init_price, gamma=gamma) # not so liquid
            # init cex
            cex = CEX( sigma=sigma, init_price=init_price) # very liquid
            # arbitrageur and price sensitive trader
            arb = Arbitrageur(x=0., y=0.)
            trader_price_sensitive = Trader(x=0., y=0.)
            # arrival process for price sensitive trader
            lag_arrival_process = sample_exponential(lam=lam)
            last_arrival_time = 0
    
            for i, t in enumerate(ts[:-1]):
                h = ts[i+1] - ts[i]
                # we update the streaming premia of illiquid CFM
                S, _,_ = cex.price_descr
                cpm_illiquid.update_streaming_premia(sigma=sigma, h=h, S=S)
                # exogenous price movement in CEX
                cex.update_S(h=h)
                # Arbitrageur makes profit (if possible)
                arb.execute_arbitrage(to_liquid = cex, to_illiquid = cpm_illiquid)
                # poisson process for price sensitive trade
                if t > (last_arrival_time + lag_arrival_process):
                    trader_price_sensitive.send_trade_price_sensitive(to_illiquid=cpm_illiquid, to_liquid=cex)
                    lag_arrival_process = sample_exponential(lam=lam)
                    last_arrival_time = t

            # save episode
            S_liquid, _, _ = cex.price_descr
            S_illiquid, _, _ = cpm_illiquid.price_descr
            sim['gamma'].append(gamma)
            sim['cex_S'].append(S_liquid)
            sim['cfm_illiquid_x'].append(cpm_illiquid.x)
            sim['cfm_illiquid_y'].append(cpm_illiquid.y)
            sim['cfm_illiquid_S'].append(S_illiquid)
            sim['cfm_illiquid_fees_x'].append(cpm_illiquid.fees.x)
            sim['cfm_illiquid_fees_y'].append(cpm_illiquid.fees.y)
            sim['cfm_illiquid_fees'].append(cpm_illiquid.fees.y + cpm_illiquid.fees.x * S_liquid)
            sim['cfm_illiquid_streaming_premia'].append(cpm_illiquid.streaming_premia)
            sim['imp_loss'].append(cpm_illiquid.y + cpm_illiquid.x * S_liquid + cpm_illiquid.fees.y + cpm_illiquid.fees.x * S_liquid - (10*init_price + 10 * S_liquid))
            sim['arb_y'].append(arb.y)
            sim['sigma'].append(sigma)
            sim['episode'].append(n_mc)
            sim['lambda'].append(lam)
        tqdm.write('lambda={}, sigma={}, gamma={:.3f}'.format(lam, sigma, gamma))

    return sim, uniswap_fees



if __name__ == '__main__':
    init_price = 10.
    
    lambdas = [75, 100]#[25,50,75]
    gammas = np.linspace(0.96,1,9)
    sigmas = [0.2,0.4,0.6,0.8]#[0.2,0.4,0.8,1.6]
    N_mc = 100
    ts = np.linspace(0,1,101)

    sim, uniswap_fees = run_sim(init_price=init_price, N_mc = N_mc, gammas=gammas, sigmas=sigmas, lambdas = lambdas)
    df = pd.DataFrame(sim)
    df.to_csv('data/sim_cfm_cex_noise_arb.csv')

    #fig, ax = plt.subplots()
    #ax.plot(df['cfm_liquid_S'], df['imp_loss'], 'o', alpha=0.5)
    #fig.savefig('data/plot_imp_loss.pdf')




