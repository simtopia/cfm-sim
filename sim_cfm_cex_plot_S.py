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
import seaborn as sns
import argparse
import os

from cfm_sim.trader import Trader, Arbitrageur
from cfm_sim.cpm import CPM
from cfm_sim.cex import CEX
from cfm_sim.exchange import Pool
from cfm_sim.poisson_process import sample_exponential


def run_sim(init_price: float, N_mc: int, gammas, sigmas, plots_dir, ts: np.ndarray = np.linspace(0,1,101)):

    sim = defaultdict(list)
    uniswap_fees = defaultdict(list)
    lam =75 
    # SIMULATE N_MC scenarios for each gamma, sigma
    for gamma, sigma in [(0.8,0.4), (0.9,0.4), (0.96, 0.4), (0.98,0.4), (0.997,0.4)]: #[0.97]: #[ 0.9,0.95,0.96,0.97,0.98,0.99,1.]:
        print('gamma = {}, sigma={}'.format(gamma, sigma))
        for n_mc in tqdm(range(N_mc)):
            np.random.seed(10101)
            # init CFM pool
            cpm_illiquid = CPM(x=10., y=10. * init_price, gamma=gamma) # not so liquid
            # init cex
            cex = CEX( sigma=sigma, init_price=init_price) # very liquid
            # init trader wallet and arb wallet
            arb = Arbitrageur(x=0., y=0.)
            trader_price_sensitive = Trader(x=0., y=0.)
            #Poisson
            lag_arrival_process = sample_exponential(lam=lam)
            last_arrival_time = 0

            fees_x = []
            fees_y = []
            fees_premium = []
            last_fees = Pool(0,0)
            S_aux = np.ones(ts.shape)*10
            h = ts[1] - ts[0]
            S_aux[1:] = np.exp(-0.5 * sigma**2 * h + sigma * np.sqrt(h)*np.random.randn(100))
            S_aux = S_aux.cumprod()
            for i, t in enumerate(ts[:-1]):
                last_premium = cpm_illiquid.streaming_premia

                h = ts[i+1] - ts[i]
                # we update the streaming premia of illiquid CFM
                S, _,_ = cex.price_descr
                cpm_illiquid.update_streaming_premia(sigma=sigma, h=h, S=S)
                # exogeneous price movement in CEX
                #cex.update_S(h=h,)
                cex.S_past.append(S_aux[i+1])
                # Arbitrageur makes profit (if possible)
                arb.execute_arbitrage(to_liquid = cex, to_illiquid = cpm_illiquid)
                
                # poisson process for price sensitive trade
                if args.poisson:
                    if t > (last_arrival_time + lag_arrival_process):
                        trader_price_sensitive.send_trade_price_sensitive(to_illiquid=cpm_illiquid, to_liquid=cex)
                        lag_arrival_process = sample_exponential(lam=lam)
                        last_arrival_time = t
                
                cpm_illiquid.update_past()
                fees_x.append((cpm_illiquid.fees.x - last_fees.x) / h)
                fees_y.append((cpm_illiquid.fees.y - last_fees.y) / h)
                fees_premium.append((cpm_illiquid.streaming_premia - last_premium)/h)
                last_fees.x = cpm_illiquid.fees.x
                last_fees.y = cpm_illiquid.fees.y

        # save episode
        cpm_S = cpm_illiquid.S_past
        cex_S = cex.S_past

        fees_y = np.array(fees_y) + np.array(fees_x) * cex_S[-1]
        fees_premium = np.array(fees_premium)


        
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(ts, cpm_S, label=r"cpm $S_t$")
        ax.plot(ts, cex_S, label=r"cex $S_t$")
        ax.set_xlabel("time")
        ax.set_title(r"$\sigma=0.4, \gamma = %.3f$" % gamma)
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(plots_dir, "S_sigma{}_gamma{}.pdf".format(sigma, gamma)))
        
        
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12,4), sharey=True)
        ax0.plot(ts[:-1], fees_y, color='steelblue')
        ax0.fill_between(ts[:-1], 0, fees_y, color='steelblue', alpha=0.5)
        ax0.set_xlabel('time')
        ax0.set_title(r"Uniswap fee, $\sigma=0.4, \gamma = %.3f$" % gamma)
        ax0.grid(True)
        textstr = 'Accumulated fees: {:.3f}'.format(fees_y.sum() * h)
        props = dict(boxstyle='round', facecolor='steelblue', alpha=0.2)
        ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, fontsize=10, verticalalignment='top', bbox=props)

        ax1.plot(ts[:-1], fees_premium, color='seagreen')
        ax1.fill_between(ts[:-1], 0, fees_premium, color='seagreen', alpha=0.5)
        ax1.set_xlabel('time')
        ax1.set_title(r"Non-arb fee, $\sigma=0.4, \gamma = %.3f$" % gamma)
        ax1.grid(True)
        textstr = 'Accumulated fees: {:.3f}'.format(fees_premium.sum() * h)
        props = dict(boxstyle='round', facecolor='seagreen', alpha=0.2)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        fig.savefig(os.path.join(plots_dir, "fees_sigma{}_gamma{}.pdf".format(sigma, gamma)))

        tqdm.write('sigma={}, gamma={:.3f}'.format(sigma, gamma))

    return sim, uniswap_fees



if __name__ == '__main__':
    init_price = 10.
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--poisson', action='store_true', default=False)
    args = parser.parse_args()
    if args.poisson:
        results_dir = os.path.join('plots', 'poisson')
    else:
        results_dir = os.path.join('plots', 'just_arb')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    gammas = np.linspace(0.97,1,7)
    sigmas = [0.2,0.4,0.8]
    N_mc = 1
    ts = np.linspace(0,1,101)

    sim, uniswap_fees = run_sim(init_price=init_price, N_mc = N_mc, gammas=gammas, sigmas=sigmas, plots_dir = results_dir)
    df = pd.DataFrame(sim)
    df.to_csv('data/sim_cfm_cex_arb.csv')




