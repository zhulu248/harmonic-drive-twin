"""
Friction-induced driving torque and the 2D -> 3D extension (canonical, shared).

Frictionless, the no-load driving torque is exactly zero: stored elastic energy is
invariant under WG rotation, so T = -dU/dphi = 0.  Friction breaks that: a WG
rotating at Omega drags its surface past the (stationary) ring; Coulomb friction mu
gives a tangential traction mu*p wherever the normal pressure is p, so

    T_drive = (1/Omega) * INTEGRAL[ mu |p| v_rel ] ds = mu R INTEGRAL|p| ds = mu R F_N

per unit axial width, with F_N = INTEGRAL|p| ds the total normal contact force.
For the two-sided cos2theta pressure p = (9 D w0/R^4) cos2theta:
    F_N = 36 D w0 / R^3   =>   T_drive = 36 mu D w0 / R^2   (per unit width)
With a flexible bearing, sliding -> rolling, mu -> mu_eff (~0.001-0.005).

2D -> 3D (uniform tube of axial width b): intensive quantities (w, strain, stress,
p) are b-independent; extensive quantities (M, V, N, F_N, T) scale linearly with b.
Plane-strain D (long tube) is ~1/(1-nu^2) stiffer than plane-stress D (free ends).
"""
import os
import sys
import numpy as np

from harmonic_drive_twin.flex_common.ring import RingParams          # noqa: E402
from harmonic_drive_twin.flex_common.ellipse import exact_fields     # noqa: E402


def normal_force_two_sided_cos2(par):
    """F_N = integral|p| ds for the pure cos2theta two-sided pressure [N per width]."""
    return 36.0 * par.D * par.w0 / par.R ** 3        # = 4 * p0 * R, p0 = 9Dw0/R^4


def normal_force_exact(par, n_max=12):
    """F_N from the exact-ellipse two-sided pressure (numerical integral of |p|)."""
    th = np.linspace(0, 2 * np.pi, 4001)
    p = exact_fields(par, th, n_max)["p"]
    trap = np.trapz if hasattr(np, "trapz") else np.trapezoid
    return trap(np.abs(p), th) * par.R


def friction_torque(par, mu, model="cos2", b=1.0, n_max=12):
    """Driving torque from friction.  Returns dict (per width and x b)."""
    if model == "cos2":
        F_N = normal_force_two_sided_cos2(par)
        T_closed = 36.0 * mu * par.D * par.w0 / par.R ** 2     # per width
    else:
        F_N = normal_force_exact(par, n_max)
        T_closed = mu * par.R * F_N
    T_per_width = mu * par.R * F_N
    return dict(model=model, mu=mu, F_N_per_width=F_N, T_per_width=T_per_width,
                T_closed_per_width=T_closed, b=b,
                F_N_total=F_N * b, T_total=T_per_width * b)


def zero_torque_check(par, n_max=12):
    """Frictionless driving torque = 0: U(phi) is flat (energy invariant)."""
    th = np.linspace(0, 2 * np.pi, 4001, endpoint=False)
    EI = par.D
    trap = np.trapz if hasattr(np, "trapz") else np.trapezoid
    Us = []
    for phi in np.linspace(0, np.pi, 7):
        M = exact_fields(par, th - phi, n_max)["M"]
        Us.append(trap(M ** 2, th) / (2 * EI) * par.R)
    Us = np.array(Us)
    return dict(U_mean=Us.mean(), U_spread=(Us.max() - Us.min()) / Us.mean())


if __name__ == "__main__":
    par = RingParams()
    for mu, lab in [(0.10, "sliding cam"), (0.003, "flexible bearing")]:
        c = friction_torque(par, mu, "cos2")
        print(f"mu={mu:<5} [{lab}]  F_N={c['F_N_per_width']:.4g} N/mm  "
              f"T_drive={c['T_per_width']:.4g} N*mm/mm")
    print("zero-torque check:", zero_torque_check(par))
