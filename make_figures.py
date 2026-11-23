from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tir_reader import load_tir
import tyre_model as tm
import vehicle as vh

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

tyre = load_tir(ROOT / "data" / "sample.tir")

# 1. Lateral force against slip angle at four loads
alpha_deg = np.linspace(-15, 15, 601)
alpha = np.radians(alpha_deg)
fig, ax = plt.subplots(figsize=(7, 4.5))
for Fz in [2000, 4000, 6000, 8000]:
    ax.plot(alpha_deg, tm.lateral_force(tyre, alpha, Fz), label=f"Fz = {Fz} N")
ax.set_xlabel("Slip angle (deg)")
ax.set_ylabel("Lateral force Fy (N)")
ax.set_title("Pure lateral force at four loads")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "lateral_force.png", dpi=150)

# 2. Friction ellipse at 4000 N
kappa = np.linspace(-1, 1, 801)
fig, ax = plt.subplots(figsize=(7, 4.5))
for a_deg in [-12, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 12]:
    Fx, Fy = tm.combined_force(tyre, kappa, np.radians(a_deg), 4000)
    ax.plot(Fx, Fy, color="tab:blue", linewidth=1)
ax.set_xlabel("Longitudinal force Fx (N)")
ax.set_ylabel("Lateral force Fy (N)")
ax.set_title("Friction ellipse at Fz = 4000 N (slip ratio swept -1 to 1)")
ax.axis("equal")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "friction_ellipse.png", dpi=150)

# 3. Axle grip against weight transfer
transfers = np.linspace(0, 3500, 15)
grip = [vh.axle_peak_force(tyre, 4000, d)[0] / 8000 for d in transfers]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(transfers, grip, marker="o")
ax.set_ylim(0, 1)
ax.set_xlabel("Weight transfer from inner to outer tyre (N)")
ax.set_ylabel("Axle peak lateral force / axle load")
ax.set_title("Weight transfer costs axle grip (8000 N axle)")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "axle_grip.png", dpi=150)

print("Saved figures to", OUT)