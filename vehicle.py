
import numpy as np
import tyre_model as tm


def axle_peak_force(tyre, Fz_static, dFz, turn="right"):
    alpha = np.radians(np.linspace(0, 20, 2001))
    if turn == "left":
        alpha = -alpha  # left turn: negative slip angles, force to the left

    Fz_outer = min(Fz_static + dFz, 2 * Fz_static)
    Fz_inner = Fz_static - dFz
    if turn == "right":
        Fz_left, Fz_right = Fz_outer, Fz_inner  # turning right, the left wheel is outside
    else:
        Fz_left, Fz_right = Fz_inner, Fz_outer

    Fy_axle = np.zeros_like(alpha)
    if Fz_left > 0:
        Fy_axle = Fy_axle + tm.lateral_force(tyre, alpha, Fz_left)
    if Fz_right > 0:
        Fy_axle = Fy_axle + tm.lateral_force_right(tyre, alpha, Fz_right)

    i = np.argmax(np.abs(Fy_axle))
    return abs(Fy_axle[i]), abs(np.degrees(alpha[i]))


def car_balance(tyre, mass, front_weight_share, cg_height, track,
                front_roll_share, ay_g=1.0):
    W = mass * 9.81
    Fz_front = W * front_weight_share / 2
    Fz_rear = W * (1 - front_weight_share) / 2

    dFz_total = W * ay_g * cg_height / track
    dFz_front = front_roll_share * dFz_total
    dFz_rear = (1 - front_roll_share) * dFz_total

    front_peak, _ = axle_peak_force(tyre, Fz_front, dFz_front)
    rear_peak, _ = axle_peak_force(tyre, Fz_rear, dFz_rear)
    return front_peak / (2 * Fz_front), rear_peak / (2 * Fz_rear)


def max_lateral_g(tyre, mass, front_weight_share, cg_height, track,
                  front_roll_share, verbose=False):
    ay = 1.0
    for step in range(50):
        mu_f, mu_r = car_balance(tyre, mass, front_weight_share, cg_height,
                                 track, front_roll_share, ay)
        new_ay = min(mu_f, mu_r)
        if verbose:
            print(step, round(ay, 4), round(mu_f, 4), round(mu_r, 4))
        if abs(new_ay - ay) < 1e-4:
            return new_ay, mu_f, mu_r
        ay = new_ay
    raise RuntimeError("Did not converge")