import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tir_reader import load_tir, write_tir
import tyre_model as tm
import fitting as ft

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
TRUE_TIR = ROOT / "data" / "sample.tir"
FITTED_TIR = ROOT / "data" / "fitted.tir"

true_tyre = load_tir(TRUE_TIR)
rng = np.random.default_rng(2026)
loads = [2000, 4000, 6000, 8000]
noise = 20.0


# ---------- synthetic test programme, measured on the true tyre ----------
a_sweep = np.radians(np.arange(-12, 12.01, 0.5))
lat_alpha = np.tile(a_sweep, len(loads))
lat_Fz = np.repeat(loads, a_sweep.size)
lat_Fy = tm.lateral_force(true_tyre, lat_alpha, lat_Fz) + rng.normal(0, noise, lat_alpha.size)

k_sweep = np.linspace(-0.3, 0.3, 61)
lon_kappa = np.tile(k_sweep, len(loads))
lon_Fz = np.repeat(loads, k_sweep.size)
lon_Fx = tm.longitudinal_force(true_tyre, lon_kappa, lon_Fz) + rng.normal(0, noise, lon_kappa.size)

angles = [-8, -4, -2, 0, 2, 4, 8]
com_kappa = np.tile(k_sweep, len(loads) * len(angles))
com_alpha = np.radians(np.tile(np.repeat(angles, k_sweep.size), len(loads)))
com_Fz = np.repeat(loads, len(angles) * k_sweep.size)
Fx_true, Fy_true = tm.combined_force(true_tyre, com_kappa, com_alpha, com_Fz)
com_Fx = Fx_true + rng.normal(0, noise, com_kappa.size)
com_Fy = Fy_true + rng.normal(0, noise, com_kappa.size)


# ---------- fit stage by stage; each stage builds on the last ----------
stages = [
    ("pure lateral", "LATERAL_COEFFICIENTS",
     ["PCY1", "PDY1", "PDY2", "PEY1", "PEY2", "PEY3", "PKY1", "PKY2", "PKY4",
      "PHY1", "PHY2", "PVY1", "PVY2"],
     [1.3, 1.0, 0, 0, 0, 0, -15.0, 1.5, 2.0, 0, 0, 0, 0],
     [0.3, 0.2, 0.2, 0.5, 0.5, 0.2, 5, 1, 0.5, 0.005, 0.005, 0.02, 0.02],
     lambda t: tm.lateral_force(t, lat_alpha, lat_Fz), lat_Fy),
    ("pure longitudinal", "LONGITUDINAL_COEFFICIENTS",
     ["PCX1", "PDX1", "PDX2", "PEX1", "PEX2", "PEX3", "PEX4",
      "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2"],
     [1.65, 1.0, 0, 0, 0, 0, 0, 20.0, 0, 0, 0, 0, 0, 0],
     [0.3, 0.2, 0.2, 0.5, 0.5, 0.5, 0.1, 10, 20, 1, 0.002, 0.002, 0.01, 0.01],
     lambda t: tm.longitudinal_force(t, lon_kappa, lon_Fz), lon_Fx),
    ("combined, Fx", "LONGITUDINAL_COEFFICIENTS",
     ["RBX1", "RBX2", "RCX1", "REX1", "REX2", "RHX1"],
     [10.0, 10.0, 1.0, 0, 0, 0],
     [5, 5, 0.3, 0.5, 0.5, 0.01],
     lambda t: tm.combined_force(t, com_kappa, com_alpha, com_Fz)[0], com_Fx),
    ("combined, Fy", "LATERAL_COEFFICIENTS",
     ["RBY1", "RBY2", "RBY3", "RCY1", "REY1", "REY2", "RHY1", "RHY2",
      "RVY1", "RVY2", "RVY4", "RVY5", "RVY6"],
     [10.0, 10.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 50.0, 2.0, 20.0],
     [5, 5, 0.01, 0.3, 0.5, 0.5, 0.01, 0.01, 0.05, 0.05, 50, 1, 15],
     lambda t: tm.combined_force(t, com_kappa, com_alpha, com_Fz)[1], com_Fy),
]

fitted_tyre = true_tyre  # coefficients not fitted here are taken from the file
updates = {}
for label, section, names, p0, spread, predict, measured in stages:
    starts = ft.random_starts(p0, spread, 10, seed=0)
    (x, std, corr, rms), n_agree, _ = ft.fit_multistart(
        fitted_tyre, section, names, starts, predict, measured)
    fitted_tyre = ft.with_coefficients(fitted_tyre, section, names, x)
    updates.setdefault(section, {}).update(dict(zip(names, x)))
    print(f"{label:18s} RMS {rms:6.2f} N   starts agreeing {n_agree}")

write_tir(TRUE_TIR, FITTED_TIR, updates)
print("Wrote", FITTED_TIR)


# ---------- check the fitted file against the independent reference ----------
fitted = load_tir(FITTED_TIR)
ref = pd.read_csv(ROOT / "data" / "mf61_reference.csv")
a, k, Fz, g = (ref[c].to_numpy() for c in ["alpha_rad", "kappa", "Fz", "gamma_rad"])

mine = {
    "Fx0": tm.longitudinal_force(fitted, k, Fz),
    "Fy0": tm.lateral_force(fitted, a, Fz, g),
}
mine["Fx"], mine["Fy"] = tm.combined_force(fitted, k, a, Fz, g)

zero_camber = g == 0
for name, values in mine.items():
    error = np.abs(values - ref[name].to_numpy())
    print(f"{name:4s} max error vs reference: {error.max():6.1f} N   "
          f"(zero camber only: {error[zero_camber].max():6.1f} N)")