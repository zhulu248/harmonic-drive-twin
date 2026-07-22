"""
Canonical baseline design point for the flexspline / harmonic-drive project.

ONE agreed set of numbers so every topic (topic1..topic5) studies the SAME part.
If a topic needs a different value, it should override locally and SAY SO in its
findings -- never silently redefine the baseline.

Values reconciled from stages 1-4 (validated) and the parametric cup script.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    # --- rim / ring (stages 1-3) ---
    R: float = 20.0          # mean rim radius                              [mm]
    t: float = 0.4           # rim wall thickness                          [mm]
    E: float = 2.0e5         # Young's modulus (steel)                     [MPa]
    nu: float = 0.30         # Poisson ratio                               [-]
    w0: float = 0.30         # WG radial ovalization at major axis         [mm]
    plane_strain: bool = True

    # --- teeth (stage 4) ---
    z_flex: int = 200        # flexspline tooth count                      [-]
    z_circ: int = 202        # circular-spline tooth count (z_flex + 2)    [-]
    alpha_deg: float = 20.0  # pressure angle                              [deg]
    T_out: float = 30.0e3    # output torque                          [N*mm] (30 N*m)

    # --- material strength (stage 4) ---
    Su: float = 1500.0       # ultimate tensile strength                   [MPa]
    Se: float = 600.0        # endurance limit                             [MPa]
    q: float = 0.80          # notch sensitivity                           [-]

    # --- cup / diaphragm geometry (stage 3 cup script) ---
    L: float = 30.0          # cup (cylinder) length; open end at z=L      [mm]
    t_d: float = 2.0         # diaphragm thickness                         [mm]
    R_hole: float = 11.0     # central bore radius (output-shaft clamp)    [mm]
    b_WG: float = 10.0       # axial width of the WG contact band          [mm]
    n_bolt: int = 12         # output-flange bolt holes in the diaphragm   [-]

    @property
    def module(self) -> float:
        """Gear module m = 2R / z_flex [mm]."""
        return 2.0 * self.R / self.z_flex

    @property
    def ratio(self) -> float:
        """Reduction ratio z_flex / (z_circ - z_flex)."""
        return self.z_flex / (self.z_circ - self.z_flex)


BASELINE = Baseline()


def ring_params():
    """Build a common.ring.RingParams at the baseline."""
    from common.ring import RingParams
    b = BASELINE
    return RingParams(R=b.R, t=b.t, E=b.E, nu=b.nu, w0=b.w0,
                      plane_strain=b.plane_strain)


def tooth_models():
    """Build (ToothGeom, LoadCase, Material) at the baseline."""
    from common.teeth import ToothGeom, LoadCase, Material
    b = BASELINE
    geom = ToothGeom(R=b.R, z=b.z_flex, alpha_deg=b.alpha_deg)
    load = LoadCase(T_out=b.T_out)
    mat = Material(Su=b.Su, Se=b.Se, q=b.q)
    return geom, load, mat


if __name__ == "__main__":
    b = BASELINE
    print(f"BASELINE  R={b.R} t={b.t} w0={b.w0}  z={b.z_flex}/{b.z_circ}  "
          f"m={b.module:.4g} mm  ratio={b.ratio:.0f}:1  T_out={b.T_out/1e3:.0f} N*m")
    print(f"cup  L={b.L} t_d={b.t_d} R_hole={b.R_hole} b_WG={b.b_WG} n_bolt={b.n_bolt}")
