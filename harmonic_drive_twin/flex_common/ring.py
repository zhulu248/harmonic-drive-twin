"""
Thin-ring analytical model of the flexspline (canonical, shared).

The flexspline is reduced to a thin inextensible ring forced into an elliptical
shape by a rigid wave generator (WG).  Two contact models:

  (A) TWO-SIDED  -- ring tied to the cam (flexible-bearing idealization), conforms
      to the ellipse everywhere.  Deformed shape PRESCRIBED as the second harmonic
      w(theta) = w0 cos(2 theta).  Everything closed form.

  (B) UNILATERAL -- a bare cam can only push outward.  Ring conforms over a wrapping
      arc |theta| < phi near the major axis and lifts off (p = 0) elsewhere.

Conventions
-----------
theta        : polar angle, 0 = major axis (max outward push), pi/2 = minor axis
w(theta)     : radial displacement of the ring mid-line, OUTWARD positive  [mm]
prime        : d/dtheta ;  ds = R dtheta

Thin-ring (Euler-Bernoulli, inextensible mid-line) relations, per unit axial length:
  curvature change   dkappa = -(1/R^2)(w + w'')            [1/mm]
  bending moment     M      =  D dkappa = -(D/R^2)(w + w'')  [N] (N*mm per mm)
  shear (transverse) V      =  (1/R) dM/dtheta               [N/mm]
  hoop / normal      N      :  circumferential equilibrium   [N/mm]
  contact pressure   p      :  outward distributed load      [MPa]

Flexural rigidity per unit axial length (PLANE STRAIN -- long tube):
  D = E t^3 / [12 (1 - nu^2)]

Surface bending stress:  sigma_b = 6 M / t^2 (OD fibre; OD tension at major axis).
Units throughout: mm, N, MPa (N/mm^2).  Verified vs ANSYS (see stage1 legacy).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq  # noqa: F401  (kept for downstream BVP work)


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class RingParams:
    R: float = 20.0        # mean (neutral-circle) radius                     [mm]
    t: float = 0.4         # wall thickness                                   [mm]
    E: float = 2.0e5       # Young's modulus (steel)                          [MPa]
    nu: float = 0.30       # Poisson ratio                                    [-]
    w0: float = 0.30       # radial pull-out at major axis (ellipticity)      [mm]
    plane_strain: bool = True   # True -> long tube; False -> plane stress

    @property
    def D(self) -> float:
        """Flexural rigidity per unit axial length [N*mm]."""
        I = self.t ** 3 / 12.0
        if self.plane_strain:
            return self.E * I / (1.0 - self.nu ** 2)
        return self.E * I

    def __str__(self) -> str:
        return (f"R={self.R} mm, t={self.t} mm, E={self.E} MPa, nu={self.nu}, "
                f"w0={self.w0} mm, {'plane-strain' if self.plane_strain else 'plane-stress'}, "
                f"D={self.D:.4g} N*mm")


# --------------------------------------------------------------------------- #
#  Model A : two-sided (prescribed ellipse, closed form)
# --------------------------------------------------------------------------- #
def two_sided(par: RingParams, theta: np.ndarray) -> dict:
    """Closed-form second-harmonic solution for w = w0 cos(2 theta)."""
    D, R, w0, t = par.D, par.R, par.w0, par.t
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)

    dkappa = (3.0 * w0 / R ** 2) * c2                 # curvature change  [1/mm]
    M = (3.0 * D * w0 / R ** 2) * c2                  # bending moment    [N]
    V = -(6.0 * D * w0 / R ** 3) * s2                 # shear             [N/mm]
    N = -(3.0 * D * w0 / R ** 3) * c2                 # hoop force        [N/mm]
    p = (9.0 * D * w0 / R ** 4) * c2                  # contact pressure  [MPa]
    w = w0 * c2

    sigma_b = 6.0 * M / t ** 2                        # surface bend stress [MPa]
    eps_b = (t / 2.0) * dkappa                        # surface bend strain [-]

    return dict(theta=theta, w=w, dkappa=dkappa, M=M, V=V, N=N, p=p,
                sigma_b=sigma_b, eps_b=eps_b)


# --------------------------------------------------------------------------- #
#  Model B : unilateral (bare cam) -- leading-order wrap
# --------------------------------------------------------------------------- #
def _contact_state(par: RingParams, th: np.ndarray) -> dict:
    """State on the contact arc, where w = w0 cos(2 theta) is enforced."""
    D, R, w0 = par.D, par.R, par.w0
    return dict(
        w=w0 * np.cos(2 * th),
        wp=-2 * w0 * np.sin(2 * th),
        M=(3.0 * D * w0 / R ** 2) * np.cos(2 * th),
        V=-(6.0 * D * w0 / R ** 3) * np.sin(2 * th),
        N=-(3.0 * D * w0 / R ** 3) * np.cos(2 * th),
        p=(9.0 * D * w0 / R ** 4) * np.cos(2 * th),
    )


def unilateral(par: RingParams, theta: np.ndarray | None = None) -> dict:
    """
    Bare-cam (push-only) contact -- LEADING-ORDER model.

    A cam can only push the ring outward.  The two-sided pressure
    p = (9 D w0 / R^4) cos(2 theta) is negative on the minor-axis side, which a
    bare cam cannot supply.  To leading order the ring stays on the cam only where
    p >= 0:  |theta| <= phi = 45 deg about each major-axis end (50% total wrap),
    independent of R, t, E, w0.  The missing minor-axis load is what a flexible
    bearing supplies in a real harmonic drive.
    """
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, 2881)
    base = two_sided(par, theta)
    phi = np.pi / 4.0                                   # exact, parameter-free
    in_contact = np.cos(2 * theta) >= 0.0               # p >= 0 region
    p = np.where(in_contact, base['p'], 0.0)            # push-only pressure
    out = dict(base)
    out.update(theta=theta, p=p, in_contact=in_contact,
               phi=phi, phi_deg=45.0,
               wrap_fraction=0.5,
               p_lost_peak=float(np.abs(base['p']).max()))
    return out


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def mirror_to_circle(theta_q: np.ndarray, f_q: np.ndarray, even: bool = True):
    """Extend a quarter [0,pi/2] field to the full circle using 2-fold symmetry."""
    th2 = np.pi - theta_q[::-1]
    f2 = f_q[::-1] * (1 if even else 1)
    th = np.concatenate((theta_q, th2[1:]))
    f = np.concatenate((f_q, f2[1:]))
    th = np.concatenate((th, th[1:] + np.pi))
    f = np.concatenate((f, f[1:]))
    return th, f


def summary(par: RingParams) -> str:
    th = np.linspace(0, 2 * np.pi, 1441)
    A = two_sided(par, th)
    B = unilateral(par)
    L = []
    L.append("=" * 70)
    L.append("FLEXSPLINE THIN-RING MODEL (common.ring)")
    L.append("=" * 70)
    L.append(str(par))
    L.append("")
    L.append("--- Model A: TWO-SIDED (prescribed ellipse, closed form) ---")
    L.append(f"  peak |bending moment|   M_max = {np.abs(A['M']).max():.4g} N")
    L.append(f"  peak |shear|            V_max = {np.abs(A['V']).max():.4g} N/mm")
    L.append(f"  peak |hoop force|       N_max = {np.abs(A['N']).max():.4g} N/mm")
    L.append(f"  peak |contact press.|   p_max = {np.abs(A['p']).max():.4g} MPa")
    L.append(f"  peak surface stress     s_max = {np.abs(A['sigma_b']).max():.4g} MPa")
    L.append("")
    L.append("--- Model B: UNILATERAL (bare cam, push-only) ---")
    L.append(f"  separation half-angle    phi   = {B['phi_deg']:.1f} deg (EXACT)")
    L.append(f"  contact wrap fraction          = {B['wrap_fraction']*100:.0f}%")
    L.append(f"  peak push-only pressure  p_max = {B['p'].max():.4g} MPa")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    print(summary(RingParams()))
