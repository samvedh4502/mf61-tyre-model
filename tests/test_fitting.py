from pathlib import Path

import numpy as np

from tir_reader import load_tir
import tyre_model as tm
import fitting as ft

ROOT = Path(__file__).parent.parent
TYRE = load_tir(ROOT / "data" / "sample.tir")

NAMES = ["PCX1", "PDX1", "PDX2", "PEX1", "PEX2", "PEX3", "PEX4",
         "PKX1", "PKX2", "PKX3", "PHX1", "PHX2", "PVX1", "PVX2"]
P0 = [1.65, 1.0, 0, 0, 0, 0, 0, 20.0, 0, 0, 0, 0, 0, 0]
SPREAD = [0.3, 0.2, 0.2, 0.5, 0.5, 0.5, 0.1, 10, 20, 1, 0.002, 0.002, 0.01, 0.01]

KAPPA = np.tile(np.linspace(-0.3, 0.3, 61), 4)
FZ = np.repeat([2000, 4000, 6000, 8000], 61)


def predict(trial):
    return tm.longitudinal_force(trial, KAPPA, FZ)


def test_noise_free_data_recovers_exact_coefficients():
    measured = predict(TYRE)
    starts = ft.random_starts(P0, SPREAD, 10, seed=0)
    (x, _, _, _), _, _ = ft.fit_multistart(TYRE, "LONGITUDINAL_COEFFICIENTS", NAMES,
                                          starts, predict, measured)
    true = np.array([TYRE["LONGITUDINAL_COEFFICIENTS"][n] for n in NAMES])
    assert np.abs(x - true).max() < 1e-6


def test_multistart_beats_a_single_generic_start():
    rng = np.random.default_rng(7)
    measured = predict(TYRE) + rng.normal(0, 20, size=KAPPA.size)
    _, _, _, rms_single = ft.fit(TYRE, "LONGITUDINAL_COEFFICIENTS", NAMES, P0, predict, measured)
    starts = ft.random_starts(P0, SPREAD, 10, seed=0)
    (_, _, _, rms_best), _, _ = ft.fit_multistart(TYRE, "LONGITUDINAL_COEFFICIENTS", NAMES,
                                                 starts, predict, measured)
    assert rms_best < rms_single - 1.0