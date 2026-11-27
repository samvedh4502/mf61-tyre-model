from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tir_reader import load_tir, write_tir
import tyre_model as tm

ROOT = Path(__file__).parent.parent
TRUE = load_tir(ROOT / "data" / "sample.tir")
FITTED = load_tir(ROOT / "data" / "fitted.tir")
REF = pd.read_csv(ROOT / "data" / "mf61_reference.csv")

A, K, FZ, G = (REF[c].to_numpy() for c in ["alpha_rad", "kappa", "Fz", "gamma_rad"])
INSIDE = (np.abs(K) <= 0.3) & (np.abs(A) <= np.radians(12))  # where the test programme measured


def forces(tyre):
    Fx, Fy = tm.combined_force(tyre, K, A, FZ, G)
    return {"Fx0": tm.longitudinal_force(tyre, K, FZ),
            "Fy0": tm.lateral_force(tyre, A, FZ, G),
            "Fx": Fx, "Fy": Fy}


TRUE_F = forces(TRUE)
FIT_F = forces(FITTED)


@pytest.mark.parametrize("name", ["Fx0", "Fy0", "Fx", "Fy"])
def test_fitted_tyre_within_noise_inside_tested_range(name):
    error = np.abs(FIT_F[name] - TRUE_F[name])
    assert error[INSIDE].max() < 20.0  # the measurement noise


def test_unfitted_coefficients_unchanged():
    assert FITTED["ALIGNING_COEFFICIENTS"] == TRUE["ALIGNING_COEFFICIENTS"]
    assert FITTED["DIMENSION"] == TRUE["DIMENSION"]


def test_write_tir_without_changes_reproduces_file(tmp_path):
    copy = tmp_path / "copy.tir"
    write_tir(ROOT / "data" / "sample.tir", copy, {})
    assert load_tir(copy) == TRUE