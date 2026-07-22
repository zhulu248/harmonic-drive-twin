"""Parameter dataclasses for the harmonic-drive digital twin.

All physical quantities use a consistent unit system throughout:

* length  — mm
* force   — N
* torque  — N·mm
* speed   — rad/s (input), rpm where noted
* temperature — °C

Note
----
``MeshParams`` is defined in ``flex_base/common/mesh.py`` and re-exported via
:mod:`harmonic_drive_twin.twin`.  This module covers the remaining four
parameter blocks: friction, thermal, tolerances, and stiffness chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


# ---------------------------------------------------------------------------
# Friction / lubrication
# ---------------------------------------------------------------------------

@dataclass
class FrictionParams:
    """Friction and lubrication parameters.

    The model separates three independent loss channels:

    1. **Tooth-flank friction** — Stribeck curve at the boundary/mixed
       lubrication regime (film ratio λ ≈ 0.1 for typical greased HD at rated
       speed; constant boundary coefficient ``mu_k`` is the dominant term).
    2. **WG bearing drag** — rolling-element (thin-section ball) bearing;
       load-dependent term follows Palmgren ``M1 = f1 P d_m`` with
       ``f1 ~ 3×10⁻⁴`` for ball bearings.
    3. **Churn + seal** — speed-dependent, load-independent channel: Palmgren
       ``M0 = 1e-7 f0 (ν n)^{2/3} d_m^3`` plus seal lip drag.

    Parameters
    ----------
    mu_s : float
        Static / boundary Stribeck coefficient [-].
    mu_k : float
        Kinetic plateau coefficient [-].
    v_str : float
        Stribeck transition velocity at reference temperature [mm/s].
    delta : float
        Stribeck exponent (2 = Gaussian) [-].
    mu_v : float
        Viscous (EHL) drag slope at reference temperature [s/mm].
    mu_wg : float
        WG flexible-bearing effective rolling friction coefficient [-].
        Palmgren ``f1 ~ 5×10⁻⁴`` for thin-section ball bearings.
    wg_breakaway : float
        Static multiplier on ``mu_wg`` at rest [-].
    f0_churn : float
        Palmgren lubrication factor for the churn channel [-].
        Calibrated to TPI 275 no-load torque; physically 2–4 for thin-section
        four-point-contact bearings with grease lubrication.
    d_m : float
        WG bearing mean diameter [mm].
    T_seal : float
        Seal lip drag at running speed [N·mm].
    seal_breakaway : float
        Static multiplier on ``T_seal`` at rest [-].
    T_adh : float
        Grease-adhesion breakaway torque [N·mm].  Scales with fill volume
        (``∝ size³``).  Calibrated to TPI 275 starting-torque row.
    sigma_rough : float
        Composite flank roughness Rq for the EHL film-ratio calculation [mm].

    Notes
    -----
    Calibration order (rev-E protocol, do not change without cause):

    1. ``T_adh`` — fit to starting-torque row (``M0 = 0`` at rest).
    2. ``f0_churn`` — LSQ fit to no-load running-torque row.
    3. ``T_seal`` kept at 5 N·mm.
    4. ``mu_wg`` kept at Palmgren ``f1 = 5×10⁻⁴``.
    5. ``mu_k`` — last, fit to efficiency row; must stay in [0.05, 0.12].

    See Also
    --------
    FrictionParams.typical : Uncalibrated literature values (no catalogue fit).
    """

    mu_s: float = 0.070
    mu_k: float = 0.050
    v_str: float = 30.0
    delta: float = 2.0
    mu_v: float = 5.0e-5
    mu_wg: float = 5.0e-4
    wg_breakaway: float = 2.0
    f0_churn: float = 3.43
    d_m: float = 34.0
    T_seal: float = 5.0
    seal_breakaway: float = 1.5
    T_adh: float = 14.3
    sigma_rough: float = 1.0e-4

    @classmethod
    def typical(cls) -> "FrictionParams":
        """Return uncalibrated literature values.

        Suitable as a starting point before catalogue fitting.  The flank
        coefficient is set to the standard greased-steel boundary value
        (mu = 0.10); the churn factor uses the lower end of the grease range
        (f0 = 1.5); adhesion breakaway is zero.

        Returns
        -------
        FrictionParams
            Parameter set with no catalogue-specific calibration.
        """
        return cls(mu_s=0.11, mu_k=0.10, v_str=30.0, delta=2.0, mu_v=5.0e-5,
                   mu_wg=5.0e-4, wg_breakaway=2.0, f0_churn=1.5, d_m=34.0,
                   T_seal=5.0, seal_breakaway=1.5, T_adh=0.0)

    def mu(self, v: "np.ndarray | float", visc_scale: float = 1.0) -> "np.ndarray":
        """Evaluate the Stribeck friction coefficient.

        Parameters
        ----------
        v : array_like
            Sliding speed [mm/s].
        visc_scale : float
            Viscosity ratio ``ν(T)/ν_ref``; shifts the Stribeck knee and
            viscous slope with temperature.

        Returns
        -------
        numpy.ndarray
            Friction coefficient at each speed.
        """
        v = np.abs(np.asarray(v, dtype=float))
        v_str_T = self.v_str / visc_scale
        mu_v_T = self.mu_v * visc_scale
        return (self.mu_k
                + (self.mu_s - self.mu_k) * np.exp(-(v / v_str_T) ** self.delta)
                + mu_v_T * v)

    def mu_wg_eff(self, v_surf: float) -> float:
        """WG bearing effective friction coefficient (with static multiplier)."""
        s = np.exp(-(abs(v_surf) / self.v_str) ** self.delta)
        return self.mu_wg * (1.0 + (self.wg_breakaway - 1.0) * s)

    def T_drag(self, omega_in: float, nu_cSt: float) -> float:
        """Churn-plus-seal drag torque, input-referred [N·mm].

        Parameters
        ----------
        omega_in : float
            WG angular speed [rad/s].  Use 0 for static (breakaway).
        nu_cSt : float
            Grease base-oil kinematic viscosity at the operating temperature
            [cSt].

        Returns
        -------
        float
            Load-independent drag torque [N·mm].
        """
        if omega_in == 0.0:
            return self.T_seal * self.seal_breakaway + self.T_adh
        n_rpm = abs(omega_in) * 60.0 / (2.0 * np.pi)
        churn = (1.0e-7 * self.f0_churn
                 * (nu_cSt * n_rpm) ** (2.0 / 3.0)
                 * self.d_m ** 3)
        return self.T_seal + churn


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------

@dataclass
class ThermalParams:
    """Temperature-dependent material and viscosity model.

    The operating temperature *T* is passed as a live input to
    :meth:`HarmonicDriveTwin.solve`, allowing measured temperatures to be
    fed in directly.

    Parameters
    ----------
    T_ref : float
        Reference temperature at which all catalogue calibrations hold [°C].
        Schaeffler TPI 275 quotes all no-load / efficiency / back-drive values
        at +20 °C.
    nu_ref : float
        Grease base-oil kinematic viscosity at *T_ref* [cSt].
    beta_visc : float
        Viscosity–temperature coefficient [1/K]: ``ν(T) = ν_ref exp(-β(T - T_ref))``.
    dT_fs_cs : float
        Flexspline-minus-circular-spline temperature split [K].  Positive = FS
        hotter; closes flank clearances.
    dT_wg_fs : float
        WG-minus-FS temperature split [K].  Affects effective ovalization.
    alpha_s : float
        Steel thermal-expansion coefficient [1/K].
    """

    T_ref: float = 20.0
    nu_ref: float = 100.0
    beta_visc: float = 0.035
    dT_fs_cs: float = 0.0
    dT_wg_fs: float = 0.0
    alpha_s: float = 11.5e-6

    def nu(self, T: float) -> float:
        """Kinematic viscosity at temperature *T* [cSt]."""
        return self.nu_ref * np.exp(-self.beta_visc * (T - self.T_ref))

    def visc_scale(self, T: float) -> float:
        """Viscosity ratio ``ν(T) / ν_ref``."""
        return float(np.exp(-self.beta_visc * (T - self.T_ref)))

    def dc_flank(self, R: float, alpha: float) -> float:
        """Flank clearance shift from FS–CS temperature split [mm]."""
        return -np.sin(alpha) * self.alpha_s * R * self.dT_fs_cs

    def dw0(self, R: float) -> float:
        """Effective ovalization change from WG–FS temperature split [mm]."""
        return self.alpha_s * R * self.dT_wg_fs


# ---------------------------------------------------------------------------
# Manufacturing tolerances
# ---------------------------------------------------------------------------

@dataclass
class ToleranceParams:
    """Per-tooth clearance model (backlash + pitch scatter).

    Parameters
    ----------
    backlash : float
        Design backlash [mm] (full gap on the line of action; each flank sees
        half).
    sigma_pitch : float
        Standard deviation of pitch-error scatter [mm].  Set to 0 for
        deterministic (all-equal) clearances.
    seed : int
        Base random seed for the tooth-clearance draw.
    clamp_negative : bool
        If ``True``, negative clearances (interference) are clamped to zero.
        The default (``False``) allows thermal preload.

    Notes
    -----
    Clearance realisations are drawn once at construction time via
    :meth:`gaps`.  Use ``seed_offset`` in :meth:`HarmonicDriveTwin.monte_carlo`
    to sweep over independent realisations.
    """

    backlash: float = 0.0
    sigma_pitch: float = 0.0
    seed: int = 0
    clamp_negative: bool = False

    def gaps(self, n: int, seed_offset: int = 0) -> "np.ndarray":
        """Draw *n* tooth clearances on the line of action [mm].

        Parameters
        ----------
        n : int
            Number of engaged teeth.
        seed_offset : int
            Added to ``self.seed``; use to generate independent Monte-Carlo
            realisations.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Per-tooth clearance values [mm].
        """
        b_h = 0.5 * self.backlash
        if self.sigma_pitch <= 0.0:
            c = np.full(n, b_h)
        else:
            rng = np.random.default_rng(self.seed + seed_offset)
            c = b_h + rng.normal(0.0, self.sigma_pitch, size=n)
        return np.maximum(0.0, c) if self.clamp_negative else c


# ---------------------------------------------------------------------------
# Stiffness chain
# ---------------------------------------------------------------------------

@dataclass
class StiffnessChain:
    """Series torsional stiffness chain, referred to the output shaft.

    The total drive compliance is the sum of three contributions in series:

    1. **Mesh** — from the force-negotiation solve (output-torque dependent).
    2. **Structure** — cup wall + diaphragm + WG bearing (``K_struct``).
    3. **Extra** — optional additional series element (default: infinite = absent).

    Parameters
    ----------
    k0 : float
        FE-anchored single-tooth mesh stiffness [N/mm].  Calibrated from
        PLANE183 analysis of the shared FSTooth geometry.
    K_struct : float
        Lumped structural torsional stiffness referred to the output [N·mm/rad].
        Default is the catalogue K1 value for size 17.
    K_extra : float
        Optional additional series stiffness [N·mm/rad].  Use ``np.inf`` to
        omit.
    use_hertz : bool
        If ``True``, the Hertz flank-contact compliance is added in series with
        ``k0`` (reduces effective stiffness by ~12 %).
    k_hertz : float
        Hertz line-contact stiffness per tooth [N/mm].

    Class Attributes
    ----------------
    K_STRUCT_FE : float
        FE-computed structural wind-up (lobe-loaded SHELL281 model) [N·mm/rad].
    K_BEARING_BR : float
        Ball-race WG bearing torsional stiffness [N·mm/rad].

    Notes
    -----
    The physically-derived chain (``StiffnessChain.physical()``) uses FE
    structural + ball-race bearing values and matches catalogue K3 within 20 %
    with zero stiffness calibration.  Compliance shares: structure 73 %,
    WG bearing 25 %, mesh 1.3 %.
    """

    k0: float = 6.85e4
    K_struct: float = 1.0e7
    K_extra: float = float(np.inf)
    use_hertz: bool = False
    k_hertz: float = 5.15e5

    K_STRUCT_FE: float = field(default=2.66e7, init=False, repr=False)
    K_BEARING_BR: float = field(default=7.67e7, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "K_STRUCT_FE", 2.66e7)
        object.__setattr__(self, "K_BEARING_BR", 7.67e7)

    @property
    def k0_eff(self) -> float:
        """Effective per-tooth stiffness (with optional Hertz series term)."""
        if self.use_hertz:
            return 1.0 / (1.0 / self.k0 + 1.0 / self.k_hertz)
        return self.k0

    @staticmethod
    def structural_decomposition(
        R: float = 20.0, t: float = 0.4, L: float = 30.0,
        t_d: float = 2.0, R_hole: float = 11.0,
        G: float = 7.69e4, K_total: float = 1.0e7,
    ) -> dict:
        """Closed-form torsional stiffness decomposition [N·mm/rad].

        Parameters
        ----------
        R, t, L : float
            Flexspline pitch radius, wall thickness, cup length [mm].
        t_d, R_hole : float
            Diaphragm thickness and bore radius [mm].
        G : float
            Shear modulus [MPa].
        K_total : float
            Total measured torsional stiffness [N·mm/rad].

        Returns
        -------
        dict
            Keys: ``K_tube``, ``K_plate``, ``K_struct_bound``, ``K_bearing``,
            ``bearing_share``.
        """
        K_tube = G * 2.0 * np.pi * R ** 3 * t / L
        K_plate = 4.0 * np.pi * G * t_d / (1.0 / R_hole ** 2 - 1.0 / R ** 2)
        K_struct_bound = 1.0 / (1.0 / K_tube + 1.0 / K_plate)
        K_bearing = 1.0 / max(1e-30, 1.0 / K_total - 1.0 / K_struct_bound)
        return dict(K_tube=K_tube, K_plate=K_plate,
                    K_struct_bound=K_struct_bound, K_bearing=K_bearing,
                    bearing_share=K_total / K_bearing)

    @classmethod
    def physical(cls, **kw) -> "StiffnessChain":
        """Physically-derived stiffness chain (FE structure + ball-race bearing).

        Returns a chain that matches catalogue K3 within 20 % with no stiffness
        calibration.

        Parameters
        ----------
        **kw
            Forwarded to the constructor (e.g. ``k0``, ``use_hertz``).

        Returns
        -------
        StiffnessChain
        """
        inst = cls(**kw)
        return replace(inst, K_struct=inst.K_STRUCT_FE, K_extra=inst.K_BEARING_BR)

    def series(self, K_mesh: float) -> "tuple[float, list[float]]":
        """Combine mesh, structural, and extra stiffnesses in series.

        Parameters
        ----------
        K_mesh : float
            Instantaneous mesh torsional stiffness [N·mm/rad].

        Returns
        -------
        K_total : float
            Combined torsional stiffness [N·mm/rad].
        components : list of float
            Individual stiffnesses [K_mesh, K_struct, K_extra].
        """
        Ks = [K_mesh, self.K_struct, self.K_extra]
        inv = sum(1.0 / K for K in Ks if np.isfinite(K) and K > 0)
        return 1.0 / inv, Ks
