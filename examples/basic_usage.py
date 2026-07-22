"""Basic usage examples for harmonic_drive_twin.

Run from the package root::

    python examples/basic_usage.py
"""
import numpy as np
from harmonic_drive_twin import (
    HarmonicDriveTwin,
    solve_point,
    twin_for_size,
    validate,
    CATALOGUE,
)

RPM = 2 * np.pi / 60

# ---------------------------------------------------------------------------
# 1. One-shot solve via the functional wrapper
# ---------------------------------------------------------------------------
print("=== 1. solve_point (one-shot) ===")
s = solve_point(T_out_Nm=30.0, speed_rpm=2000, T_degC=20.0)
print(f"  η = {s['eta']*100:.1f} %   T_in = {s['T_in']:.0f} N·mm")
print(f"  loss split:  flanks {s['T_teeth_in']:.1f}  |  "
      f"WG bearing {s['T_wg_in']:.1f}  |  churn+seal {s['T_churn']:.1f}  N·mm")

# ---------------------------------------------------------------------------
# 2. Catalogue-preset twin (size 17)
# ---------------------------------------------------------------------------
print("\n=== 2. Catalogue-preset twin (size 17) ===")
tw17 = twin_for_size(17)
s17 = tw17.solve(T_out=31_000, omega_in=2000 * RPM, T=20.0)
print(f"  η = {s17['eta']*100:.1f} %  (catalogue: 77 ± 3 %)")
print(f"  Back-drive threshold: {tw17.backdrive_threshold(T=20.0)/1e3:.2f} N·m  "
      f"(catalogue: 3.06 N·m)")
print(f"  Starting input torque: {tw17.breakaway_input_torque(T=20.0):.1f} mN·m  "
      f"(catalogue: 29 mN·m)")

# ---------------------------------------------------------------------------
# 3. Full catalogue validation table
# ---------------------------------------------------------------------------
print("\n=== 3. Full catalogue validation ===")
rows = validate()
print(f"  {'size':>4}  {'η_twin':>7}  {'η_cat':>7}  {'no-load':>9}  {'cat':>7}  {'start':>7}  {'cat':>7}")
for r in rows:
    print(f"  {r['size']:>4}  {r['eta']:>7.1f}  {r['eta_cat']:>7.0f}  "
          f"{r['T_nlrt']:>9.1f}  {r['T_nlrt_cat']:>7.0f}  "
          f"{r['T_nlst']:>7.1f}  {r['T_nlst_cat']:>7.0f}")

# ---------------------------------------------------------------------------
# 4. Efficiency curve (torque sweep)
# ---------------------------------------------------------------------------
print("\n=== 4. Efficiency curve (torque sweep at 2000 rpm) ===")
tw = HarmonicDriveTwin()
T_range = np.linspace(5_000, 60_000, 12)
curve = tw.efficiency_curve(T_range, 2000 * RPM, T=20.0)
for T_Nm, eta in zip(curve["T"] / 1e3, curve["eta"]):
    bar = "█" * int(eta * 40)
    print(f"  {T_Nm:5.1f} N·m  {eta*100:5.1f} %  {bar}")

# ---------------------------------------------------------------------------
# 5. Thermal warm-up (30-minute transient)
# ---------------------------------------------------------------------------
print("\n=== 5. Thermal warm-up (30 min at rated) ===")
wu = tw.warmup(T_out=30_000, omega_in=2000 * RPM, t_end=1800)
print(f"  Steady-state temperature : {wu['T_ss']:.1f} °C")
print(f"  Steady-state η           : {wu['eta_ss']*100:.1f} %")
print(f"  Power loss at steady state: {wu['P_loss_ss']:.2f} W")
print(f"  Grease life estimate      : {wu['L_grease_h']:.0f} h")

# ---------------------------------------------------------------------------
# 6. Hertz / EHL contact check
# ---------------------------------------------------------------------------
print("\n=== 6. Hertz / EHL at rated ===")
hz = tw.hertz_flank_summary()
print(f"  p_max = {hz['p_max']:.0f} MPa   a = {hz['a']*1e3:.2f} µm")
print(f"  h_min = {hz['h_min']*1e6:.1f} nm   λ = {hz['lam']:.2f} (boundary/mixed)")
