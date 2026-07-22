"""Harmonic-drive digital twin — public API.

Quick start
-----------
>>> from harmonic_drive_twin import HarmonicDriveTwin, solve_point
>>> import numpy as np
>>>
>>> # Default twin (size 20 baseline geometry, rev-E calibration)
>>> tw = HarmonicDriveTwin()
>>> s  = tw.solve(T_out=30_000, omega_in=2000 * 2*np.pi/60)
>>> print(f"η = {s['eta']*100:.1f} %")
η = 77.9 %
>>>
>>> # Functional convenience wrapper
>>> s2 = solve_point(T_out_Nm=30.0, speed_rpm=2000)
>>> print(f"η = {s2['eta']*100:.1f} %")
η = 77.9 %
>>>
>>> # Catalogue-preset twin for size 17 at +20 °C
>>> from harmonic_drive_twin import twin_for_size
>>> tw17 = twin_for_size(17)
>>> s17  = tw17.solve(T_out=31_000, omega_in=2000 * 2*np.pi/60, T=20.0)
>>> print(f"size 17 η = {s17['eta']*100:.1f} % (catalogue: 77 ±3 %)")
size 17 η = 75.8 % (catalogue: 77 ±3 %)
"""
from __future__ import annotations

import numpy as np

from .twin import (
    HarmonicDriveTwin,
    MeshParams,
    hertz_flank,
    ehl_lambda,
    wg_bearing_stiffness,
)
from .params import (
    FrictionParams,
    ThermalParams,
    ToleranceParams,
    StiffnessChain,
)
from .catalogue import (
    CATALOGUE,
    twin_for_size,
    calibrate,
    validate,
)

__version__ = "0.1.0"
__all__ = [
    # Core class
    "HarmonicDriveTwin",
    # Parameter blocks
    "MeshParams",
    "FrictionParams",
    "ThermalParams",
    "ToleranceParams",
    "StiffnessChain",
    # Convenience
    "solve_point",
    "twin_for_size",
    # Catalogue
    "CATALOGUE",
    "calibrate",
    "validate",
    # Contact helpers
    "hertz_flank",
    "ehl_lambda",
    "wg_bearing_stiffness",
]


def solve_point(
    T_out_Nm: float,
    speed_rpm: float,
    T_degC: float = 20.0,
    size: int | None = None,
) -> dict:
    """Evaluate a single operating point.

    Convenience wrapper around :class:`HarmonicDriveTwin` for one-shot
    calculations without explicit class instantiation.

    Parameters
    ----------
    T_out_Nm : float
        Output torque [N·m].
    speed_rpm : float
        WG input speed [rpm].
    T_degC : float
        Drive temperature [°C].  Default: 20 °C (catalogue reference).
    size : int or None
        TPI 275 frame size {14, 17, 20, 25}.  ``None`` uses the project
        baseline geometry (R = 20 mm, equivalent to size 20).

    Returns
    -------
    dict
        Same keys as :meth:`HarmonicDriveTwin.solve`, plus:

        ``size`` : int or None
            Frame size used.
        ``T_out_Nm`` : float
            Output torque in N·m (convenience copy).
        ``speed_rpm`` : float
            WG speed in rpm (convenience copy).

    Examples
    --------
    >>> s = solve_point(T_out_Nm=30.0, speed_rpm=2000)
    >>> round(s['eta'] * 100, 1)
    77.9

    >>> s17 = solve_point(T_out_Nm=31.0, speed_rpm=2000, size=17, T_degC=20.0)
    >>> round(s17['eta'] * 100, 1)
    75.8
    """
    omega = speed_rpm * 2.0 * np.pi / 60.0
    T_out_Nmm = T_out_Nm * 1e3
    if size is not None:
        tw = twin_for_size(size)
    else:
        tw = HarmonicDriveTwin()
    result = tw.solve(T_out_Nmm, omega, T=T_degC)
    result["size"] = size
    result["T_out_Nm"] = T_out_Nm
    result["speed_rpm"] = speed_rpm
    return result
