from scipy import stats
import math
import numpy as np
__author__ = 'MagnieAr'


def black_scholes(call_put, spot, strike, t, vol, rf, div):
        """
        rf: risk-free rate
        div: dividend
        call_put: +1/-1 for call/put
        """

        d1 = (math.log(spot/strike)+(rf-div+0.5*math.pow(vol, 2))*t)/(vol*math.sqrt(t))
        d2 = d1 - vol*math.sqrt(t)

        option_price = (call_put*spot*math.exp(-div*t)*stats.norm.cdf(call_put*d1)) - \
                       (call_put*strike*math.exp(-rf*t)*stats.norm.cdf(call_put*d2))

        return option_price;


def volatility_model(coefficient_matrix, tau):

    vol_1 = 0
    vol_2 = 0
    vol_3 = 0

    total_vol = np.sqrt(vol_1 + vol_2 + vol_3)

    return total_vol;


def first_sum():




    result = 0

    return result;


def second_sum():





    result = 0

    return result;

