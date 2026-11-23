from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tir_reader import load_tir
import tyre_model as tm

ROOT = Path(__file__).parent.parent
TYRE = load_tir(ROOT / "data" / "sample.tir")
REF = pd.read_csv(ROOT / "data" / "mf61_reference.csv")


def model_outputs():
    out = {"Fx0": [], "Fy0": [], "Mz0": [], "Fx": [], "Fy": [], "Mz": []}
    for _, row in REF.iterrows():
        a, k, Fz, g = row["alpha_rad"], row["kappa"], row["Fz"], row["gamma_rad"]
        out["Fx0"].append(tm.longitudinal_force(TYRE, k, Fz))
        out["Fy0"].append(tm.lateral_force(TYRE, a, Fz, g))
        out["Mz0"].append(tm.aligning_moment(TYRE, a, Fz, gamma=g)[0])
        Fx, Fy = tm.combined_force(TYRE, k, a, Fz, g)
        out["Fx"].append(Fx)
        out["Fy"].append(Fy)
        out["Mz"].append(tm.aligning_moment(TYRE, a, Fz, kappa=k, gamma=g)[0])
    return {name: np.array(values) for name, values in out.items()}


MINE = model_outputs()


# Largest allowed difference from the independent MF 6.1 library.
# Each tolerance is tied to a documented convention difference.
TOLERANCES = {
    "Fx0": 1e-3,   # identical
    "Fy0": 20.0,   # reference uses alpha, not tan(alpha), in pure Fy only
    "Mz0": 4.0,    # reference: extra cos(alpha), no s*Fx in pure Mz
    "Fx": 1e-3,    # identical
    "Fy": 20.0,    # follows from the pure Fy difference
    "Mz": 4.0,     # as Mz0
}


@pytest.mark.parametrize("name", list(TOLERANCES))
def test_matches_reference(name):
    error = np.abs(MINE[name] - REF[name].to_numpy())
    assert error.max() < TOLERANCES[name]


def test_known_values():
    assert tm.lateral_force(TYRE, np.radians(4), 4000) == pytest.approx(-2879.65, abs=0.01)
    assert tm.longitudinal_force(TYRE, -0.12, 4000) == pytest.approx(-4168.42, abs=0.01)
    Mz, trail = tm.aligning_moment(TYRE, np.radians(3), 4000)
    assert Mz == pytest.approx(44.70, abs=0.01)
    assert trail == pytest.approx(0.01834, abs=1e-5)