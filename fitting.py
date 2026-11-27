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



def random_starts(p0, spread, n, seed=0):
    rng = np.random.default_rng(seed)
    p0 = np.asarray(p0, dtype=float)
    spread = np.asarray(spread, dtype=float)
    starts = [p0]
    for _ in range(n - 1):
        starts.append(p0 + rng.uniform(-1, 1, size=p0.size) * spread)
    return starts


def fit_multistart(tyre, section, names, starts, predict, measured):
    results = []
    for start in starts:
        try:
            results.append(fit(tyre, section, names, start, predict, measured))
        except (np.linalg.LinAlgError, ValueError):
            continue  # a start that fails is skipped
    rms_values = np.array([r[3] for r in results])
    best = results[int(np.argmin(rms_values))]
    n_agree = int(np.sum(rms_values <= best[3] * 1.001 + 1e-9))
    return best, n_agree, rms_values