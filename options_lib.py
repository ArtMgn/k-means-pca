from scipy import stats
import math
import numpy as np
__author__ = 'MagnieAr'


def options_pricing(fitted_coefficients, all_returns):

    T = 10
    commodity_number = 14
    pc_number = 3
    component_vol = np.zeros(3)

    spot = 100
    strike = 100

    returns = np.array(all_returns)

    for j in range(0, commodity_number):
        cov = np.cov(all_returns[j].T)
        for t in range(1, T):  # time-to-maturity
            for pc in range(0, pc_number):
                component_vol[pc] = volatility_model(fitted_coefficients[j][pc], T-t)
            print("commo {0} vol, Maturity {1}".format(j, t))
            print(component_vol)
            all_components = component_vol[0] + component_vol[1] + component_vol[2]
            modelled_implied_vol = np.sqrt(all_components)  # utiliser le cours de vol

            ret = returns[j].T[t]
            ret_squared = [i**2 for i in ret]
            sum_ret_squared = np.sum(ret_squared)
            historical_vol = np.std(ret)  # TODO : annualized
            realized_variance = np.sqrt(12 * sum_ret_squared)

            print modelled_implied_vol, historical_vol, realized_variance, np.sqrt(cov[t][t])

            c = black(1, spot, strike, 0, T-1, T, modelled_implied_vol, 0.10, 0.10)
            print c
            print("")

    return;


def black_scholes(call_put, spot, strike, t, vol, rf, div):

    d1 = (math.log(spot/strike)+(rf-div+0.5*math.pow(vol, 2))*t)/(vol*math.sqrt(t))
    d2 = d1 - vol*math.sqrt(t)

    option_price = (call_put*spot*math.exp(-div*t)*stats.norm.cdf(call_put*d1)) - \
                   (call_put*strike*math.exp(-rf*t)*stats.norm.cdf(call_put*d2))

    return option_price;


def black(call_put, spot, strike, t, option_maturity, future_maturity, vol, rf, convenience_yield):

    future_price = spot * math.exp((rf - convenience_yield) * (future_maturity - t))
    zero_coupon = math.exp(-rf * (option_maturity - t))

    d1 = (math.log(future_price/strike)+(0.5 * math.pow(vol, 2)))/vol
    d2 = d1 - vol

    option_price = call_put * zero_coupon * \
        (future_price * stats.norm.cdf(call_put * d1) - \
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


def simulation():

    # generate several strikes

    return


def hitorical_vol():

    return;