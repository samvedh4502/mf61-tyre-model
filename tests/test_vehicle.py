from pathlib import Path

import numpy as np
import pytest

from tir_reader import load_tir
import vehicle as vh

ROOT = Path(__file__).parent.parent
TYRE = load_tir(ROOT / "data" / "sample.tir")


def test_axle_known_values():
    force, angle = vh.axle_peak_force(TYRE, 4000, 0)
    assert force == pytest.approx(7026.84, abs=0.01)
    assert angle == pytest.approx(9.02, abs=0.01)

    force, angle = vh.axle_peak_force(TYRE, 4000, 2000)
    assert force == pytest.approx(6763.82, abs=0.01)
    assert angle == pytest.approx(10.34, abs=0.01)


def test_left_and_right_turns_match():
    # with the right tyre mirrored, the car is symmetric
    for dFz in [0, 2000, 3500]:
        right, _ = vh.axle_peak_force(TYRE, 4000, dFz, turn="right")
        left, _ = vh.axle_peak_force(TYRE, 4000, dFz, turn="left")
        assert left == pytest.approx(right)


def test_lifted_inner_wheel():
    # Once the inner wheel lifts, more transfer cannot change anything
    lifted, _ = vh.axle_peak_force(TYRE, 3200, 3200)
    beyond, _ = vh.axle_peak_force(TYRE, 3200, 4000)
    assert beyond == pytest.approx(lifted)


def test_weight_transfer_costs_grip():
    transfers = [0, 1000, 2000, 3000]
    forces = [vh.axle_peak_force(TYRE, 4000, d)[0] for d in transfers]
    assert all(np.diff(forces) < 0)


def test_balance_known_values():
    mu_f, mu_r = vh.car_balance(TYRE, 1631, 0.5, 0.5, 1.5, 0.6)
    assert mu_f == pytest.approx(0.8123, abs=1e-4)
    assert mu_r == pytest.approx(0.8422, abs=1e-4)


def test_balanced_car_corners_hardest():
    balanced = vh.max_lateral_g(TYRE, 1631, 0.5, 0.5, 1.5, 0.5)[0]
    front_heavy_roll = vh.max_lateral_g(TYRE, 1631, 0.5, 0.5, 1.5, 0.6)[0]
    rear_heavy_roll = vh.max_lateral_g(TYRE, 1631, 0.5, 0.5, 1.5, 0.4)[0]
    assert balanced == pytest.approx(0.8396, abs=1e-4)
    assert balanced > front_heavy_roll
    assert balanced > rear_heavy_roll


def test_lower_cg_corners_harder():
    low = vh.max_lateral_g(TYRE, 1631, 0.5, 0.3, 1.5, 0.5)[0]
    high = vh.max_lateral_g(TYRE, 1631, 0.5, 0.6, 1.5, 0.5)[0]
    assert low > high