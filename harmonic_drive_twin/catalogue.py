"""Catalogue presets and validation against Schaeffler TPI 275.

This module provides:

* :func:`twin_for_size` — instantiate a twin pre-scaled to a catalogue size.
* :func:`calibrate` — derive the three calibration dials from the catalogue
  table (rev-E protocol).
* :func:`validate` — run the full catalogue validation and return a results
  table.
* :data:`CATALOGUE` — digitised TPI 275 data (series RT1-H-CS, i = 100,
  +20 °C).

References
----------
Schaeffler TPI 275, series RT1-H-CS, ratio i = 100.  All values at +20 °C.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .params import FrictionParams, StiffnessChain, ThermalParams
from .twin import HarmonicDriveTwin, MeshParams

# ---------------------------------------------------------------------------
# Digitised catalogue table
# ---------------------------------------------------------------------------

#: TPI 275 data (series RT1-H-CS, i = 100, all at +20 °C).
#:
#: Keys
#: ----
#: size : array_like of int
#:     Frame sizes [14, 17, 20, 25].
#: R : array_like of float
#:     Pitch radius [mm].
#: T_N : array_like of float
#:     Rated output torque [N·m].
#: eta : array_like of float
#:     Efficiency at rated torque [%] (±3 % scatter in catalogue).
#: T_nlrt : array_like of float
#:     No-load running torque at 2000 rpm [mN·m].
#: T_nlst : array_like of float
#:     Starting (no-load breakaway) torque [mN·m].
#: T_bt : array_like of float
#:     Back-driving torque [N·m].
#: K1, K3 : array_like of float
#:     Torsional stiffness at low and high torque [N·m/rad].
CATALOGUE: dict = dict(
    size=np.array([14, 17, 20, 25]),
    R=np.array([17.78, 21.59, 25.40, 31.75]),
    T_N=np.array([10.0, 31.0, 52.0, 87.0]),
    eta=np.array([67.0, 77.0, 77.0, 77.0]),
    T_nlrt=np.array([35.0, 51.0, 105.0, 195.0]),
    T_nlst=np.array([21.0, 29.0, 37.0, 69.0]),
    T_bt=np.array([2.21, 3.06, 3.89, 7.26]),
    K1=np.array([4.7e3, 10.0e3, 16.0e3, 31.0e3]),
    K3=np.array([7.1e3, 16.0e3, 29.0e3, 57.0e3]),
)

_RPM = 2.0 * np.pi / 60.0
_R_BASE = 20.0   # project baseline pitch radius [mm]


def twin_for_size(
    size: int,
    fric: FrictionParams | None = None,
) -> HarmonicDriveTwin:
    """Instantiate a twin pre-scaled to a TPI 275 catalogue size.

    All geometric parameters are linearly scaled from the project baseline
    (R = 20 mm, size 17 ≈ 21.59 mm) using the pitch-radius ratio
    ``s = R / R_base``.  Torsional stiffnesses scale as ``s³`` (FE-anchored
    values); tooth stiffness scales as ``s`` (2-D invariance × face width).

    Parameters
    ----------
    size : int
        TPI 275 frame size.  Must be one of {14, 17, 20, 25}.
    fric : FrictionParams or None
        Friction parameters.  ``None`` uses the rev-E calibrated defaults
        (``FrictionParams()``).

    Returns
    -------
    HarmonicDriveTwin
        Scaled and ready-to-use twin instance.

    Raises
    ------
    ValueError
        If *size* is not a recognised catalogue size.

    Examples
    --------
    >>> tw = twin_for_size(17)
    >>> s = tw.solve(T_out=31_000, omega_in=2000 * 2*np.pi/60, T=20.0)
    >>> round(s['eta'] * 100, 1)
    75.8
    """
    cat = CATALOGUE
    idx = np.where(cat["size"] == size)[0]
    if idx.size == 0:
        raise ValueError(f"size {size} not in catalogue {cat['size'].tolist()}")
    R = float(cat["R"][idx[0]])
    T_N = float(cat["T_N"][idx[0]])
    if fric is None:
        fric = FrictionParams()
    return _twin_for_R(R, fric)


def _twin_for_R(R: float, fric: FrictionParams) -> HarmonicDriveTwin:
    """Internal: build a geometrically scaled twin at pitch radius R."""
    s = R / _R_BASE
    par = MeshParams(R=R, w0=0.30 * s)
    fr = replace(fric, d_m=34.0 * s, T_seal=fric.T_seal * s,
                 T_adh=fric.T_adh * s ** 3)
    ch = StiffnessChain(
        k0=6.85e4 * s,
        K_struct=StiffnessChain.K_STRUCT_FE * s ** 3,
        K_extra=StiffnessChain.K_BEARING_BR * s ** 3,
    )
    th = ThermalParams()
    return HarmonicDriveTwin(par=par, fric=fr, chain=ch, thermal=th,
                             b_face=10.0 * s, L_cup=30.0 * s)


def calibrate(verbose: bool = False) -> FrictionParams:
    """Derive the three TPI 275 calibration dials (rev-E protocol).

    Starts from uncalibrated literature values (:meth:`FrictionParams.typical`)
    and adjusts three dials in a fixed, physics-ordered sequence:

    1. ``T_adh`` — grease-adhesion breakaway, fit to the starting-torque row.
    2. ``f0_churn`` — Palmgren lubrication factor, LSQ fit to the no-load row.
    3. ``mu_k`` — boundary flank friction, mean-error fit to the efficiency row
       (must stay in [0.05, 0.12]; ``mu_s = mu_k + 0.02``).

    All other parameters (``T_seal``, ``mu_wg``, ``mu_v``, …) remain at their
    literature / Palmgren values.

    Parameters
    ----------
    verbose : bool
        If ``True``, print calibration results to stdout.

    Returns
    -------
    FrictionParams
        Calibrated parameter set.

    Notes
    -----
    The calibration result is deterministic.  The default :class:`FrictionParams`
    already contains the pre-computed values; call this function to verify or
    re-derive them.
    """
    cat = CATALOGUE
    fric = FrictionParams.typical()
    om = 2000 * _RPM

    # Dial 1: T_adh on starting row
    resid_fixed, coef = [], []
    f_probe = replace(fric, T_adh=0.0)
    for R, st_cat in zip(cat["R"], cat["T_nlst"]):
        s = R / _R_BASE
        base = _twin_for_R(R, f_probe).breakaway_input_torque(T=20.0)
        resid_fixed.append(st_cat - base)
        coef.append(s ** 3)
    T_adh = max(0.0, float(np.dot(coef, resid_fixed) / np.dot(coef, coef)))
    fric = replace(fric, T_adh=T_adh)

    # Dial 2: f0_churn on no-load row
    resid_fixed, coef = [], []
    for R, nl_cat in zip(cat["R"], cat["T_nlrt"]):
        base = _twin_for_R(R, replace(fric, f0_churn=0.0)).solve(0.0, om, T=20.0)["T_in"]
        churn1 = _twin_for_R(R, replace(fric, f0_churn=1.0)).solve(0.0, om, T=20.0)["T_in"] - base
        resid_fixed.append(nl_cat - base)
        coef.append(churn1)
    f0 = float(np.dot(coef, resid_fixed) / np.dot(coef, coef))
    fric = replace(fric, f0_churn=f0)

    # Dial 3: mu_k on efficiency row (boundary-banded)
    lo, hi = 0.050, 0.120
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        f_try = replace(fric, mu_k=mid, mu_s=mid + 0.02)
        errs = [_twin_for_R(R, f_try).solve(T * 1e3, om, T=20.0)["eta"] * 100 - e
                for R, T, e in zip(cat["R"], cat["T_N"], cat["eta"])]
        if np.mean(errs) > 0.0:
            lo = mid
        else:
            hi = mid
    mu_k = 0.5 * (lo + hi)
    fric = replace(fric, mu_k=mu_k, mu_s=mu_k + 0.02)

    if verbose:
        print(f"T_adh = {fric.T_adh:.1f} N.mm × s³")
        print(f"f0    = {fric.f0_churn:.2f}  (thin-section range 2–4)")
        print(f"mu_k  = {fric.mu_k:.3f},  mu_s = {fric.mu_s:.3f}")

    return fric


def validate(fric: FrictionParams | None = None) -> list[dict]:
    """Run the full catalogue validation and return per-size results.

    Parameters
    ----------
    fric : FrictionParams or None
        Friction parameters.  ``None`` triggers :func:`calibrate` internally.

    Returns
    -------
    list of dict
        One entry per catalogue size.  Keys: ``size``, ``R``, ``eta``,
        ``eta_cat``, ``T_nlrt``, ``T_nlrt_cat``, ``T_nlst``, ``T_nlst_cat``,
        ``T_bt``, ``T_bt_cat``, ``K`` (twin torsional stiffness [N·m/rad]).

    Examples
    --------
    >>> rows = validate()
    >>> for r in rows:
    ...     print(f"size {r['size']:2d}: η = {r['eta']:.1f} % vs {r['eta_cat']:.0f} %")
    size 14: η = 67.4 % vs 67 %
    size 17: η = 75.8 % vs 77 %
    size 20: η = 76.4 % vs 77 %
    size 25: η = 74.8 % vs 77 %
    """
    cat = CATALOGUE
    if fric is None:
        fric = calibrate()
    om = 2000 * _RPM
    rows = []
    for k, (sz, R, T_N) in enumerate(zip(cat["size"], cat["R"], cat["T_N"])):
        tw = _twin_for_R(R, fric)
        s = tw.solve(T_N * 1e3, om, T=20.0)
        nl = tw.solve(0.0, om, T=20.0)
        st = tw.breakaway_input_torque(T=20.0)
        rows.append(dict(
            size=int(sz), R=R,
            eta=s["eta"] * 100.0,       eta_cat=float(cat["eta"][k]),
            T_nlrt=nl["T_in"],          T_nlrt_cat=float(cat["T_nlrt"][k]),
            T_nlst=st,                  T_nlst_cat=float(cat["T_nlst"][k]),
            T_bt=tw.backdrive_threshold(T=20.0) / 1e3,
            T_bt_lo=st * tw.N / 1e3,   T_bt_cat=float(cat["T_bt"][k]),
            K=s["K_total"] / 1e3,
        ))
    return rows
