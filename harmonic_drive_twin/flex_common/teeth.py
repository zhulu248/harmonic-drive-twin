"""
Closed-form tooth-root stress of the flexspline (canonical, shared).

Parametric surrogate for the max tooth-root fillet stress of a strain-wave
flexspline (Lewis + Dolan-Broghamer first approximation, to be anchored by FE):

  sigma_root(theta) = sigma_dev(theta)          [ovalization / WG bending, common.ring]
                    + Kf * sigma_nom(W_t(theta)) [torque-transmitted tooth bending]

  Nominal (Lewis cantilever):   sigma_nom = 6 W_t h_F / (b s_F^2)
  Fillet concentration (Dolan-Broghamer 1942):  Kt = H + (s_F/rho_F)^L (s_F/h_F)^M
  Fatigue notch factor:         Kf = 1 + q (Kt - 1)

Cyclic combination at the critical tooth (major axis): ovalization fully reversed,
torque pulse in phase with the +ovalization peak ->
  sigma_alt = sigma_dev + sigma_tooth/2 ,  sigma_mean = sigma_tooth/2 .
Goodman equivalent fully-reversed:  sigma_ar = sigma_alt / (1 - sigma_mean/Su).

Units: mm, N, MPa.  theta = 0 is the major axis.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import numpy as np

from harmonic_drive_twin.flex_common.ring import RingParams          # noqa: E402


# Dolan-Broghamer fillet stress-concentration constants by pressure angle.
_DB_CONSTANTS = {          # alpha[deg] : (H, L, M)
    14.5: (0.22, 0.20, 0.40),
    20.0: (0.18, 0.15, 0.45),
    25.0: (0.14, 0.11, 0.50),
}


@dataclass
class ToothGeom:
    """Flexspline tooth geometry.  Shape set by ratios to the module so SIZE
    (module) and SHAPE (ratios) vary independently."""
    R: float = 20.0           # pitch / mean-rim radius                         [mm]
    z: int = 200              # number of flexspline teeth                      [-]
    b: float = 10.0           # face width                                      [mm]
    alpha_deg: float = 20.0   # pressure angle                                  [deg]
    sF_ratio: float = 1.50    # root chord thickness  s_F = sF_ratio * m        [-]
    hF_ratio: float = 2.00    # load height (moment arm) h_F = hF_ratio * m     [-]
    rhoF_ratio: float = 0.38  # root fillet radius  rho_F = rhoF_ratio * m      [-]
    kt_cal: float = 1.00      # FE calibration multiplier on Kt                 [-]
    krim: float = 1.00        # rim-ovalization concentration at the root       [-]

    @property
    def m(self) -> float:
        return 2.0 * self.R / self.z

    @property
    def s_F(self) -> float:
        return self.sF_ratio * self.m

    @property
    def h_F(self) -> float:
        return self.hF_ratio * self.m

    @property
    def rho_F(self) -> float:
        return self.rhoF_ratio * self.m

    @property
    def pitch(self) -> float:
        return np.pi * self.m

    def Kt(self) -> float:
        H, L, Mc = _db_constants(self.alpha_deg)
        return self.kt_cal * (H + (self.s_F / self.rho_F) ** L * (self.s_F / self.h_F) ** Mc)

    def Kf(self, q: float = 0.80) -> float:
        return 1.0 + q * (self.Kt() - 1.0)


@dataclass
class LoadCase:
    """Output-torque load and how it is shared among engaged teeth.

    model='stage2' (default): tooth load W_t(theta) ~ cos2theta over |theta| < 45 deg
    engagement arcs; summing to T_out gives a PARAMETER-FREE peak W_peak = pi T_out/(R z).
    model='uniform': legacy W_peak = K_conc (T_out/R)/(engage_fraction z).
    """
    T_out: float = 30.0e3        # output torque                        [N*mm] (=30 N*m)
    model: str = "stage2"
    engage_fraction: float = 0.30
    K_conc: float = 1.5708       # cos2theta peak/avg -> pi/2
    wrap_half_angle_deg: float = 45.0

    def tooth_load(self, geom: ToothGeom) -> dict:
        F_t_total = self.T_out / geom.R
        if self.model == "stage2":
            W_peak = np.pi * self.T_out / (geom.R * geom.z)
            n_eng = max(1.0, geom.z * (2.0 * self.wrap_half_angle_deg) / 180.0)
            W_avg = F_t_total / n_eng
        else:
            n_eng = max(1.0, self.engage_fraction * geom.z)
            W_avg = F_t_total / n_eng
            W_peak = self.K_conc * W_avg
        return dict(F_t_total=F_t_total, n_eng=n_eng, W_avg=W_avg, W_peak=W_peak)


@dataclass
class Material:
    Su: float = 1500.0    # ultimate tensile strength (HD flexspline steel)  [MPa]
    Se: float = 600.0     # fully-reversed endurance limit                   [MPa]
    q:  float = 0.80      # notch sensitivity                                [-]


def _db_constants(alpha_deg: float):
    keys = sorted(_DB_CONSTANTS)
    if alpha_deg <= keys[0]:
        return _DB_CONSTANTS[keys[0]]
    if alpha_deg >= keys[-1]:
        return _DB_CONSTANTS[keys[-1]]
    H = np.interp(alpha_deg, keys, [_DB_CONSTANTS[k][0] for k in keys])
    L = np.interp(alpha_deg, keys, [_DB_CONSTANTS[k][1] for k in keys])
    M = np.interp(alpha_deg, keys, [_DB_CONSTANTS[k][2] for k in keys])
    return float(H), float(L), float(M)


def sigma_dev_amplitude(ring: RingParams) -> float:
    """Ovalization (WG) bending-stress amplitude at the rim OD / tooth-root base.
    sigma = 6 M_max / t^2 = 18 D w0 / (R^2 t^2).  Fully reversed at 2x WG."""
    return 18.0 * ring.D * ring.w0 / (ring.R ** 2 * ring.t ** 2)


def sigma_nom_tooth(geom: ToothGeom, load: LoadCase) -> float:
    """Nominal Lewis cantilever bending stress from the peak tooth load."""
    W = load.tooth_load(geom)["W_peak"]
    return 6.0 * W * geom.h_F / (geom.b * geom.s_F ** 2)


def root_stress(geom: ToothGeom, load: LoadCase, ring: RingParams,
                mat: Material) -> dict:
    """Full closed-form combination + Goodman equivalent at the critical tooth."""
    s_dev = sigma_dev_amplitude(ring)
    Kt = geom.Kt()
    Kf = geom.Kf(mat.q)
    s_nom = sigma_nom_tooth(geom, load)
    s_tooth = Kf * s_nom

    s_dev_root = geom.krim * s_dev
    s_max = s_dev_root + s_tooth
    s_min = -s_dev_root
    s_alt = 0.5 * (s_max - s_min)
    s_mean = 0.5 * (s_max + s_min)

    def goodman(s_a, s_m):
        return s_a / max(1e-9, 1.0 - s_m / mat.Su)

    s_ar = goodman(s_alt, s_mean)
    n_fatigue = mat.Se / s_ar if s_ar > 0 else np.inf
    n_static = mat.Su / s_max if s_max > 0 else np.inf

    s_rim_ar = goodman(s_dev, 0.0)
    n_rim = mat.Se / s_rim_ar if s_rim_ar > 0 else np.inf

    s_overall = max(s_ar, s_rim_ar)
    n_overall = min(n_fatigue, n_rim)
    governing = "tooth_root" if s_ar >= s_rim_ar else "rim"

    return dict(
        sigma_dev=s_dev, sigma_nom=s_nom, Kt=Kt, Kf=Kf, sigma_tooth=s_tooth,
        sigma_max=s_max, sigma_min=s_min, sigma_alt=s_alt, sigma_mean=s_mean,
        sigma_ar=s_ar, n_fatigue=n_fatigue, n_static=n_static,
        sigma_rim_ar=s_rim_ar, n_rim=n_rim,
        sigma_overall=s_overall, n_overall=n_overall, governing=governing,
        **{f"load_{k}": v for k, v in load.tooth_load(geom).items()},
        m=geom.m, s_F=geom.s_F, h_F=geom.h_F, rho_F=geom.rho_F,
    )


if __name__ == "__main__":
    geom, load, ring, mat = ToothGeom(), LoadCase(), RingParams(), Material()
    r = root_stress(geom, load, ring, mat)
    print(f"s_dev={r['sigma_dev']:.1f}  s_tooth={r['sigma_tooth']:.1f}  "
          f"Kt={r['Kt']:.3f}  overall={r['sigma_overall']:.1f} MPa "
          f"({r['governing']}, SF {r['n_overall']:.2f})")
