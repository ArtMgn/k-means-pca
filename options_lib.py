from scipy import stats
import data_getter as dg
import math
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

__author__ = 'MagnieAr'


def options_pricing(fitted_coefficients, all_returns, plot_surfaces):

    T = 10
    commodity_number = 14
    pc_number = 3
    component_vol = np.zeros(3)

    spot = 2.062
    strike = 2
    rf = 1

    returns = np.array(all_returns)

    for j in range(0, commodity_number):
        call_term_structure = []
        forwards = []
        strikes = dg.get_strikes(j)
        lv = []
        for t in range(0, T):  # time-to-maturity
            tau = T-t
            for pc in range(0, pc_number):
                component_vol[pc] = volatility_model(fitted_coefficients[j][pc], tau)
            print("commo {0} vol, Maturity {1}".format(j, tau))
            print(component_vol)
            all_components = component_vol[0] + component_vol[1] + component_vol[2]
            modelled_local_vol = np.sqrt(all_components)
            # print modelled_local_vol
            lv.append(modelled_local_vol)

            fwd, name = dg.get_forward_curve(j, tau-1)
            forwards.append(float(fwd[0][1]))
            """
            cov = np.cov(all_returns[j].T)
            ret = returns[j].T[t]
            ret_squared = [i**2 for i in ret]
            sum_ret_squared = np.sum(ret_squared)
            historical_vol = np.std(ret)
            realized_variance = np.sqrt(12 * sum_ret_squared)
            print modelled_implied_vol, historical_vol, realized_variance, np.sqrt(cov[t][t])
            """

            call_strikes = []
            for k in strikes:
                if k != "-":
                    c = black(1, fwd[0][1], k, 0, tau, T, modelled_local_vol, 0.1, 0.10)
                    call_strikes.append(c)
                    # print c #, call.impliedVolatility
            call_term_structure.append(call_strikes)
        print(lv)
        if plot_surfaces:
            plot_price_surface(call_term_structure, strikes, name, lv, forwards)

    return;


def black_scholes(call_put, spot, strike, t, vol, rf, div):

    d1 = (math.log(spot/strike)+(rf-div+0.5*math.pow(vol, 2))*t)/(vol*math.sqrt(t))
    d2 = d1 - vol*math.sqrt(t)

    option_price = (call_put*spot*math.exp(-div*t)*stats.norm.cdf(call_put*d1)) - \
                   (call_put*strike*math.exp(-rf*t)*stats.norm.cdf(call_put*d2))

    return option_price;


def black(call_put, forward, strike, t, option_maturity, future_maturity, local_vol, rf, convenience_yield):
    # print(call_put, forward, strike, t, option_maturity, future_maturity, local_vol, rf, convenience_yield)

    #future_price = spot * math.exp((rf - convenience_yield) * (future_maturity - t))
    zero_coupon = math.exp(-rf * (future_maturity - t))

    d1 = (math.log(forward/strike)+(0.5 * math.pow(local_vol, 2)))/local_vol
    d2 = d1 - local_vol

    option_price = call_put * zero_coupon * \
        (forward * stats.norm.cdf(call_put * d1) - \
         strike * stats.norm.cdf(call_put * d2))

    return option_price;


def volatility_model(fitted_coefficients, future_maturity):

    b = 0
    v = 0
    for i in range(0, 5):
        a = math.pow(fitted_coefficients[i], 2)*math.pow(future_maturity, 2*i+1)*(1/(2*i + 1))
        for j in range(i, 5):
            b += fitted_coefficients[i]*fitted_coefficients[j]*math.pow(future_maturity, 2*i+1)*(1/(j + i + 1))
        v += a + 2*b
        b = 0

    return v;


def plot_price_surface(call_term_structure, strikes, name, lv, forwards):

    ts = np.asarray(call_term_structure)
    k = np.asarray(strikes[:len(strikes)-1])
    fw = np.asarray(forwards[::-1])
    vol = np.asarray(lv)
    fig = plt.figure()

    fig.suptitle('Local vol model {0}'.format(name))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    # plot call prices

    ax = fig.add_subplot(gs[0], projection='3d')
    X = [i for i in range(0, 10)]
    Y = [float(j) for j in k]
    X, Y = np.meshgrid(X, Y)
    Z = ts.T

    ax.set_xlabel('Time-to-maturity')
    ax.set_ylabel('Strikes')
    ax.set_zlabel('Call prices')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plot forward curve

    ax = fig.add_subplot(gs[1])
    ax.grid(True)
    ax.set_xlabel('Time-to-maturity')
    ax.set_ylabel('Forward prices')
    l = ax.plot(X[0], fw)

    # plot local vols
    ax = fig.add_subplot(gs[2])
    ax.grid(True)
    ax.set_ylabel('Local Vols')
    ax.set_xlabel('Time-to-maturity')
    l = ax.plot(X[0], vol)

    plt.show()

    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)

    return;