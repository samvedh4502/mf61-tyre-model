import numpy as np


def magic_formula(x, B, C, D, E):
    Bx = B * x
    return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))


def weight(x, B, C, E, SH):
    def g(s):
        Bs = B * s
        return np.cos(C * np.arctan(Bs - E * (Bs - np.arctan(Bs))))
    return g(x + SH) / g(SH)


def lateral_force(tyre, alpha, Fz, gamma=0.0):
    lat = tyre["LATERAL_COEFFICIENTS"]
    Fz0 = tyre["VERTICAL"]["FNOMIN"]
    dfz = (Fz - Fz0) / Fz0
    g = np.sin(gamma)

    Cy = lat["PCY1"]
    mu_y = (lat["PDY1"] + lat["PDY2"] * dfz) * (1 - lat["PDY3"] * g**2)
    Dy = mu_y * Fz

    Kya = (lat["PKY1"] * Fz0 * (1 - lat["PKY3"] * abs(g))
           * np.sin(lat["PKY4"] * np.arctan(Fz / ((lat["PKY2"] + lat["PKY5"] * g**2) * Fz0))))
    By = Kya / (Cy * Dy)

    Kyg = Fz * (lat["PKY6"] + lat["PKY7"] * dfz)
    SVyg = Fz * (lat["PVY3"] + lat["PVY4"] * dfz) * g
    SVy = Fz * (lat["PVY1"] + lat["PVY2"] * dfz) + SVyg
    SHy = lat["PHY1"] + lat["PHY2"] * dfz + (Kyg * g - SVyg) / Kya

    alpha_y = np.tan(alpha) + SHy
    Ey = ((lat["PEY1"] + lat["PEY2"] * dfz)
          * (1 + lat["PEY5"] * g**2 - (lat["PEY3"] + lat["PEY4"] * g) * np.sign(alpha_y)))
    Ey = np.minimum(Ey, 1.0)

    return magic_formula(alpha_y, By, Cy, Dy, Ey) + SVy


def longitudinal_force(tyre, kappa, Fz):
    lon = tyre["LONGITUDINAL_COEFFICIENTS"]
    Fz0 = tyre["VERTICAL"]["FNOMIN"]
    dfz = (Fz - Fz0) / Fz0

    Cx = lon["PCX1"]
    mu_x = lon["PDX1"] + lon["PDX2"] * dfz
    Dx = mu_x * Fz

    Kxk = Fz * (lon["PKX1"] + lon["PKX2"] * dfz) * np.exp(lon["PKX3"] * dfz)
    Bx = Kxk / (Cx * Dx)

    SHx = lon["PHX1"] + lon["PHX2"] * dfz
    SVx = Fz * (lon["PVX1"] + lon["PVX2"] * dfz)

    kappa_x = kappa + SHx
    Ex = ((lon["PEX1"] + lon["PEX2"] * dfz + lon["PEX3"] * dfz**2)
          * (1 - lon["PEX4"] * np.sign(kappa_x)))
    Ex = np.minimum(Ex, 1.0)

    return magic_formula(kappa_x, Bx, Cx, Dx, Ex) + SVx


def induced_side_force(tyre, kappa, alpha, Fz, gamma=0.0):
    lat = tyre["LATERAL_COEFFICIENTS"]
    Fz0 = tyre["VERTICAL"]["FNOMIN"]
    dfz = (Fz - Fz0) / Fz0

    mu_y = lat["PDY1"] + lat["PDY2"] * dfz
    DVyk = (mu_y * Fz * (lat["RVY1"] + lat["RVY2"] * dfz + lat["RVY3"] * np.sin(gamma))
                        * np.cos(np.arctan(lat["RVY4"] * np.tan(alpha))))
    return DVyk * np.sin(lat["RVY5"] * np.arctan(lat["RVY6"] * kappa))


def combined_force(tyre, kappa, alpha, Fz, gamma=0.0):
    lat = tyre["LATERAL_COEFFICIENTS"]
    lon = tyre["LONGITUDINAL_COEFFICIENTS"]
    Fz0 = tyre["VERTICAL"]["FNOMIN"]
    dfz = (Fz - Fz0) / Fz0

    Fx0 = longitudinal_force(tyre, kappa, Fz)
    Fy0 = lateral_force(tyre, alpha, Fz, gamma)

    Bxa = lon["RBX1"] * np.cos(np.arctan(lon["RBX2"] * kappa))
    Exa = min(lon["REX1"] + lon["REX2"] * dfz, 1.0)
    Gx = weight(np.tan(alpha), Bxa, lon["RCX1"], Exa, lon["RHX1"])
    
    Byk = lat["RBY1"] * np.cos(np.arctan(lat["RBY2"] * (np.tan(alpha) - lat["RBY3"])))
    Eyk = min(lat["REY1"] + lat["REY2"] * dfz, 1.0)
    SHyk = lat["RHY1"] + lat["RHY2"] * dfz
    Gy = weight(kappa, Byk, lat["RCY1"], Eyk, SHyk)

    return Gx * Fx0, Gy * Fy0 + induced_side_force(tyre, kappa, alpha, Fz, gamma)


def aligning_moment(tyre, alpha, Fz, kappa=0.0, gamma=0.0):
    lat = tyre["LATERAL_COEFFICIENTS"]
    lon = tyre["LONGITUDINAL_COEFFICIENTS"]
    ali = tyre["ALIGNING_COEFFICIENTS"]
    Fz0 = tyre["VERTICAL"]["FNOMIN"]
    R0 = tyre["DIMENSION"]["UNLOADED_RADIUS"]
    dfz = (Fz - Fz0) / Fz0
    g = np.sin(gamma)

    # lateral quantities at zero camber (MF 6.1 convention for Mz)
    Cy = lat["PCY1"]
    Dy = (lat["PDY1"] + lat["PDY2"] * dfz) * Fz
    Kya = lat["PKY1"] * Fz0 * np.sin(lat["PKY4"] * np.arctan(Fz / (lat["PKY2"] * Fz0)))
    By = Kya / (Cy * Dy)
    SVy = Fz * (lat["PVY1"] + lat["PVY2"] * dfz)
    SHy = lat["PHY1"] + lat["PHY2"] * dfz

    # longitudinal slip stiffness
    Kxk = Fz * (lon["PKX1"] + lon["PKX2"] * dfz) * np.exp(lon["PKX3"] * dfz)

    # forces: real (cambered) ones for the lever arm, zero-camber Fy for the trail
    Fx, Fy = combined_force(tyre, kappa, alpha, Fz, gamma)
    Fx_0, Fy_0 = combined_force(tyre, kappa, alpha, Fz, 0.0)
    Fy_prime = Fy_0 - induced_side_force(tyre, kappa, alpha, Fz, 0.0)

    # slip angles for the trail and the residual torque, then their equivalents
    alpha_t = np.tan(alpha) + ali["QHZ1"] + ali["QHZ2"] * dfz + (ali["QHZ3"] + ali["QHZ4"] * dfz) * g
    alpha_r = np.tan(alpha) + SHy + SVy / Kya
    extra = (Kxk / Kya) * kappa
    alpha_t_eq = np.sqrt(alpha_t**2 + extra**2) * np.sign(alpha_t)
    alpha_r_eq = np.sqrt(alpha_r**2 + extra**2) * np.sign(alpha_r)

    # pneumatic trail
    Bt = ((ali["QBZ1"] + ali["QBZ2"] * dfz + ali["QBZ3"] * dfz**2)
          * (1 + ali["QBZ4"] * g + ali["QBZ5"] * abs(g)))
    Ct = ali["QCZ1"]
    Dt = (Fz * (R0 / Fz0) * (ali["QDZ1"] + ali["QDZ2"] * dfz)
          * (1 + ali["QDZ3"] * abs(g) + ali["QDZ4"] * g**2))
    Et = ((ali["QEZ1"] + ali["QEZ2"] * dfz + ali["QEZ3"] * dfz**2)
          * (1 + (ali["QEZ4"] + ali["QEZ5"] * g) * (2 / np.pi) * np.arctan(Bt * Ct * alpha_t)))
    Et = np.minimum(Et, 1.0)
    Bat = Bt * alpha_t_eq
    trail = Dt * np.cos(Ct * np.arctan(Bat - Et * (Bat - np.arctan(Bat)))) * np.cos(alpha)

    # residual torque, including the camber moment (QDZ8, QDZ9)
    Br = ali["QBZ9"] + ali["QBZ10"] * By * Cy
    Dr = (Fz * R0 * ((ali["QDZ6"] + ali["QDZ7"] * dfz)
                     + ((ali["QDZ8"] + ali["QDZ9"] * dfz)
                        + (ali["QDZ10"] + ali["QDZ11"] * dfz) * abs(g)) * g)
          * np.cos(alpha))
    Mzr = Dr * np.cos(np.arctan(Br * alpha_r_eq))

    # lever arm of the longitudinal force
    s = R0 * (ali["SSZ1"] + ali["SSZ2"] * (Fy / Fz0) + (ali["SSZ3"] + ali["SSZ4"] * dfz) * g)

    return -trail * Fy_prime + Mzr + s * Fx, trail

def lateral_force_right(tyre, alpha, Fz, gamma=0.0):
    # the right-hand tyre is the mirror image of the left-hand tyre in the file
    return -lateral_force(tyre, -alpha, Fz, -gamma)