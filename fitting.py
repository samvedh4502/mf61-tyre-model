import copy

import numpy as np
from scipy.optimize import least_squares


def with_coefficients(tyre, section, names, values):
    trial = copy.deepcopy(tyre)
    for name, value in zip(names, values):
        trial[section][name] = value
    return trial


def fit(tyre, section, names, p0, predict, measured):
    def residuals(p):
        return predict(with_coefficients(tyre, section, names, p)) - measured

    result = least_squares(residuals, p0)

    J = result.jac
    n_points, n_params = J.shape
    s2 = np.sum(result.fun**2) / (n_points - n_params)
    cov = np.linalg.inv(J.T @ J) * s2
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    rms = np.sqrt(np.mean(result.fun**2))
    return result.x, std, corr, rms