import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pca_lib import plot_pc


def function_fitting(fitting, function_type, eig_vectors, eig_values, sht_idx, sheet_name, plot_fitting):

    n = 10
    x = np.linspace(1, 10, n)

    if fitting == "eigen_vectors":
        v = eig_vectors.T
    else:
        d = np.diag(np.sqrt(eig_values[0:3]))
        v = eig_vectors.T.dot(d)

    fitted_curve = np.zeros((3, 10))

    #print(v.T)
    #plot_pc(v.T, sheet_name, sht_idx, 220, False)
    function_type = 0

    all_coefficients = []

    for i in range(0, 3):
        y = v.T[i, 0:10]
        if function_type == 0:
            popt, pcov = curve_fit(polynomial_func, x, y)
            fitted_curve[i] = [polynomial_func(j, popt[0], popt[1], popt[2], popt[3], popt[4]) for j in range(1, 11)]
        if function_type == 1:
            popt, pcov = curve_fit(cosine_function, x, y)
            t = [cosine_function(j, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6]) for j in range(1, 11)]
            fitted_curve[i] = t
        all_coefficients.append(popt)

    if plot_fitting:
        min = np.min(v)
        max = np.max(v)
        plt.figure()
        plt.plot(v, label='Data')
        plt.plot(fitted_curve.T, label='Fitted Data', marker='o')
        plt.ylim([min - 0.05, max + 0.05])
        plt.xlim([1, 10])
        plt.xlabel('Time-to-maturity')
        plt.ylabel('Eigen Vectors - Loading factors')
        plt.legend()
        plt.title('Volatility shape fitting - ' + sheet_name)
        plt.show()

    # Return coefficients
    return all_coefficients, v;


def polynomial_func(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + d*x**4 + e*x**5;


def cosine_function(x, a, b, c, d, e, w1, w2):
    return a + b*x + c*x**2 + d*x**3 + d*np.cos(2*np.pi*x/w1) + e*np.cos(2*np.pi*x/w2);