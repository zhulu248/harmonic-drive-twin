"""
Exact elliptical wave generator via harmonic superposition (canonical, shared).

A real elliptical WG is not a pure cos2theta: it also contains cos4theta,
cos6theta, ...  Thin-ring bending is LINEAR, so decompose the exact ellipse into
harmonics and add the per-harmonic ring responses.

Per-harmonic two-sided response (n >= 2), D = E t^3/[12(1-nu^2)]:
    w_n  = w_n cos n theta
    M_n  =  D (n^2-1)   w_n / R^2  cos n theta
    V_n  = -D n(n^2-1)  w_n / R^3  sin n theta
    N_n  = -D (n^2-1)   w_n / R^3  cos n theta
    p_n  =  D (n^2-1)^2 w_n / R^4  cos n theta
    sigma_OD_n = 6 M_n / t^2
(For n = 2 these reduce to the cos2theta formulas with 3, 6, 9 coefficients.)

Ellipse geometry: semi-major a = R + w0; semi-minor b by PERIMETER CONSERVATION
(inextensible neutral line 2 pi R) using the exact elliptic-integral perimeter.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from scipy.special import ellipe
from scipy.optimize import brentq

# import the shared RingParams from the common package
from harmonic_drive_twin.flex_common.ring import RingParams  # noqa: E402


def ellipse_perimeter(a: float, b: float) -> float:
    """Exact perimeter of an ellipse with semi-axes a >= b."""
    a, b = max(a, b), min(a, b)
    e2 = 1.0 - (b / a) ** 2
    return 4.0 * a * ellipe(e2)            # ellipe takes m = e^2


def ellipse_axes(R: float, w0: float, mode: str = "perimeter") -> tuple[float, float]:
    """Semi-axes (a, b) of the neutral-line ellipse.
    a = R + w0 (major-axis push-out). b set by:
      'perimeter' -> length conserved = 2 pi R (inextensible)   [default]
      'symmetric' -> b = R - w0 (small-deflection assumption)
    """
    a = R + w0
    if mode == "symmetric":
        return a, R - w0
    target = 2.0 * np.pi * R
    f = lambda b: ellipse_perimeter(a, b) - target
    b = brentq(f, 0.5 * R, a)
    return a, b


def ellipse_harmonics(R: float, w0: float, n_max: int = 10,
                      mode: str = "perimeter", n_grid: int = 4096) -> dict:
    """Fourier-cosine decomposition of w(theta) = r_ellipse(theta) - R.
    Returns even harmonics w_n for n = 0,2,4,...,n_max."""
    a, b = ellipse_axes(R, w0, mode)
    th = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    w = r - R
    ns = list(range(0, n_max + 1, 2))
    wn = {}
    for n in ns:
        if n == 0:
            wn[0] = np.mean(w)
        else:
            wn[n] = 2.0 * np.mean(w * np.cos(n * th))
    return dict(a=a, b=b, perimeter=ellipse_perimeter(a, b), wn=wn, ns=ns,
                theta=th, w=w, mode=mode)


def exact_fields(par: RingParams, theta: np.ndarray, n_max: int = 10,
                 mode: str = "perimeter") -> dict:
    """Total ring fields from the exact ellipse (sum of bending harmonics n>=2)."""
    D, R, t = par.D, par.R, par.t
    h = ellipse_harmonics(R, par.w0, n_max, mode)
    M = np.zeros_like(theta); V = np.zeros_like(theta)
    N = np.zeros_like(theta); p = np.zeros_like(theta); w = np.zeros_like(theta)
    per_n = {}
    for n in h["ns"]:
        if n == 0:
            continue
        wn = h["wn"][n]
        cn, sn = np.cos(n * theta), np.sin(n * theta)
        Mn = D * (n ** 2 - 1) * wn / R ** 2 * cn
        Vn = -D * n * (n ** 2 - 1) * wn / R ** 3 * sn
        Nn = -D * (n ** 2 - 1) * wn / R ** 3 * cn
        pn = D * (n ** 2 - 1) ** 2 * wn / R ** 4 * cn
        M += Mn; V += Vn; N += Nn; p += pn; w += wn * cn
        per_n[n] = dict(wn=wn, M_amp=D * (n ** 2 - 1) * wn / R ** 2,
                        p_amp=D * (n ** 2 - 1) ** 2 * wn / R ** 4,
                        sig_amp=6 * D * (n ** 2 - 1) * wn / R ** 2 / t ** 2)
    sigma_OD = 6 * M / t ** 2
    return dict(theta=theta, w=w, M=M, V=V, N=N, p=p, sigma_OD=sigma_OD,
                harmonics=h, per_n=per_n)


def cos2_fields(par: RingParams, theta: np.ndarray) -> dict:
    """Pure cos2theta model, for comparison (same w0 as major push)."""
    D, R, t, w0 = par.D, par.R, par.t, par.w0
    c2 = np.cos(2 * theta)
    M = 3 * D * w0 / R ** 2 * c2
    return dict(M=M, p=9 * D * w0 / R ** 4 * c2, sigma_OD=6 * M / t ** 2)


if __name__ == "__main__":
    par = RingParams()
    th = np.linspace(0, 2 * np.pi, 2001)
    ex = exact_fields(par, th); c2 = cos2_fields(par, th)
    print("exact vs cos2 peak |sigma_OD|:",
          f"{np.abs(ex['sigma_OD']).max():.4g} vs {np.abs(c2['sigma_OD']).max():.4g} MPa")
