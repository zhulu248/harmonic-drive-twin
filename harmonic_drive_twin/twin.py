"""Core harmonic-drive digital-twin class.

This module provides :class:`HarmonicDriveTwin`, the central physics model.
All public methods accept SI-consistent units (N, mm, rad, s, °C) unless
noted otherwise.

References
----------
Validated against Schaeffler TPI 275 (series RT1-H-CS, ratio i = 100,
+20 °C): efficiency, no-load torque, starting torque, back-driving torque,
and torsional stiffness across four catalogue sizes.  See
:mod:`harmonic_drive_twin.catalogue` for the validation script.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

from .flex_common.mesh import MeshParams, mesh_state
from .flex_common.profile import sliding_speed_base
from .flex_common.ring import RingParams
from .flex_common.friction import normal_force_two_sided_cos2
from .flex_common.baseline import BASELINE

from .params import (                                   # noqa: E402
    FrictionParams, ThermalParams, ToleranceParams, StiffnessChain,
)

# Re-export so callers can do: from harmonic_drive_twin.twin import MeshParams
__all__ = [
    "HarmonicDriveTwin", "MeshParams",
    "hertz_flank", "ehl_lambda", "wg_bearing_stiffness",
]


# ---------------------------------------------------------------------------
# Hertz / EHL helpers
# ---------------------------------------------------------------------------

def hertz_flank(
    f_tooth: float,
    b_face: float = 10.0,
    rho_tip: float = 0.06,
    R_cs: float = 0.0826,
    E: float = 2.0e5,
    nu: float = 0.30,
) -> dict:
    """Hertz line-contact state at the tooth flank.

    Parameters
    ----------
    f_tooth : float
        Normal force on one tooth [N].
    b_face : float
        Tooth face width [mm].
    rho_tip : float
        FS tip arc radius [mm].
    R_cs : float
        Conjugate CS concave arc radius [mm] (fitted from ``common.profile``
        conjugate; rms fit error < 0.5 µm).
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio [-].

    Returns
    -------
    dict
        Keys: ``Rx`` [mm], ``p_max`` [MPa], ``a`` [mm],
        ``delta`` [mm], ``k_hertz`` [N/mm].
    """
    Rx = 1.0 / (1.0 / rho_tip - 1.0 / R_cs)
    Ep = E / (1.0 - nu ** 2)
    wl = max(1e-9, f_tooth) / b_face
    p_max = np.sqrt(wl * Ep / (np.pi * Rx))
    a = np.sqrt(4.0 * wl * Rx / (np.pi * Ep))
    delta = (2.0 * wl / (np.pi * Ep)) * (np.log(4.0 * Rx / a) + 0.5)
    k_h = f_tooth / delta if delta > 0 else np.inf
    return dict(Rx=Rx, p_max=p_max, a=a, delta=delta, k_hertz=k_h)


def ehl_lambda(
    v_slide: float,
    nu_cSt: float = 100.0,
    rho_oil: float = 900.0,
    Rx: float = 0.219,
    w_line: float = 2.5,
    E: float = 2.0e5,
    nu: float = 0.30,
    sigma_rough: float = 1.0e-4,
) -> dict:
    """Dowson–Higginson EHL film thickness and lambda ratio.

    Parameters
    ----------
    v_slide : float
        Sliding speed [mm/s].
    nu_cSt : float
        Kinematic viscosity [cSt].
    rho_oil : float
        Oil density [kg/m³].
    Rx : float
        Equivalent contact radius [mm].
    w_line : float
        Line load [N/mm].
    E, nu : float
        Elastic constants of the tooth material.
    sigma_rough : float
        Composite surface roughness Rq [mm].

    Returns
    -------
    dict
        Keys: ``h_min`` [mm], ``lam`` [-], ``U``, ``W``, ``G``.
    """
    eta0 = nu_cSt * 1e-6 * rho_oil
    ue = 0.5 * abs(v_slide) / 1e3
    Rxm = Rx / 1e3
    Ep = E / (1.0 - nu ** 2) * 1e6
    w_si = w_line * 1e3
    U = eta0 * ue / (Ep * Rxm)
    W = w_si / (Ep * Rxm)
    G = 2.2e-8 * Ep
    H = 2.65 * U ** 0.7 * G ** 0.54 * W ** (-0.13) if ue > 0 else 0.0
    h_min = H * Rxm * 1e3
    return dict(h_min=h_min, lam=h_min / sigma_rough, U=U, W=W, G=G)


def wg_bearing_stiffness(
    n_b: int = 30,
    d_b: float = 1.6,
    t_race: float = 0.8,
    b_brg: float = 6.0,
    R: float = 20.0,
    t_rim: float = 0.4,
    F_r: float = 546.0,
    alpha_deg: float = 20.0,
    E: float = 2.0e5,
) -> dict:
    """WG thin-section bearing torsional stiffness (Hertz + race + rim in series).

    Parameters
    ----------
    n_b : int
        Number of balls.
    d_b, t_race : float
        Ball diameter and race thickness [mm].
    b_brg : float
        Bearing width [mm].
    R, t_rim : float
        Flexspline pitch radius and rim thickness [mm].
    F_r : float
        Radial mesh force [N].
    alpha_deg : float
        Pressure angle [°].
    E : float
        Young's modulus [MPa].

    Returns
    -------
    dict
        Keys: ``k_ball``, ``k_race``, ``k_rim``, ``k_series``, ``k_r``,
        ``K_out`` [N·mm/rad], ``Q_max``, ``R_ball``.
    """
    R_ball = R - t_rim - t_race - d_b / 2.0
    Q_max = 4.37 * F_r / n_b
    dlt = 2.0 * 2.79e-4 * Q_max ** (2.0 / 3.0) / d_b ** (1.0 / 3.0)
    k_ball = 1.5 * Q_max / dlt
    s = 2.0 * np.pi * R_ball / n_b
    k_race = 48.0 * E * (b_brg * t_race ** 3 / 12.0) / s ** 3
    k_rim = 48.0 * E * (b_brg * t_rim ** 3 / 12.0) / s ** 3
    k_ser = 1.0 / (1.0 / k_ball + 1.0 / k_race + 1.0 / k_rim)
    k_r = 2.0 * k_ser * n_b / (2.0 * 4.37)
    ta = np.tan(np.deg2rad(alpha_deg))
    K_out = R ** 2 * k_r / ta ** 2
    return dict(k_ball=k_ball, k_race=k_race, k_rim=k_rim, k_series=k_ser,
                k_r=k_r, K_out=K_out, Q_max=Q_max, R_ball=R_ball)


# ---------------------------------------------------------------------------
# Main twin class
# ---------------------------------------------------------------------------

@dataclass
class HarmonicDriveTwin:
    """Physics-based digital twin of a strain-wave (harmonic) gearbox.

    The twin computes steady-state and quasi-static quantities for a given
    operating point ``(T_out, omega_in, T)``.  It composes validated physics
    from the full ``flex_base`` FEA campaign:

    * Tooth kinematics and force negotiation (statically indeterminate mesh).
    * Hertz flank contact and EHL film analysis.
    * Stribeck tooth-flank friction (boundary / mixed regime, λ ≈ 0.1).
    * Palmgren WG-bearing rolling drag.
    * Palmgren churn + seal load-free drag.
    * Structural torsional stiffness chain (tube + diaphragm + WG bearing).
    * Thermal viscosity and thermal-expansion effects.
    * Dahl / LuGre presliding hysteresis models.
    * Archard wear-life simulation.

    Parameters
    ----------
    par : MeshParams
        Gear geometry (pitch radius, tooth counts, ovalization, …).
    fric : FrictionParams
        Friction and lubrication parameters.
    tol : ToleranceParams
        Per-tooth clearance model (backlash + pitch scatter).
    chain : StiffnessChain
        Torsional stiffness chain (mesh + structural + optional extra).
    thermal : ThermalParams
        Viscosity–temperature model and thermal-expansion coefficients.
    b_face : float
        Tooth face width / WG band width [mm].
    L_cup : float
        Cup length [mm] (used for structural decomposition).
    phi : float
        Wave-generator snapshot angle [rad].
    seed_offset : int
        Tolerance-realisation selector for Monte-Carlo sweeps.

    Examples
    --------
    Basic solve at rated conditions:

    >>> from harmonic_drive_twin import HarmonicDriveTwin
    >>> import numpy as np
    >>> tw = HarmonicDriveTwin()
    >>> result = tw.solve(T_out=30_000, omega_in=2000 * 2*np.pi/60)
    >>> print(f"η = {result['eta']*100:.1f} %")
    η = 77.9 %

    See Also
    --------
    harmonic_drive_twin.catalogue.twin_for_size : Catalogue-preset constructor.
    harmonic_drive_twin.solve_point : Functional convenience wrapper.
    """

    par: MeshParams = field(default_factory=MeshParams)
    fric: FrictionParams = field(default_factory=FrictionParams)
    tol: ToleranceParams = field(default_factory=ToleranceParams)
    chain: StiffnessChain = field(default_factory=StiffnessChain)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    b_face: float = BASELINE.b_WG
    L_cup: float = BASELINE.L
    phi: float = 0.0
    seed_offset: int = 0

    def __post_init__(self):
        st = mesh_state(self.phi, self.par.kinematic_psi_f(self.phi), self.par)
        m = st["radial_engaged"]
        self.theta = st["theta"][m]
        self.g = np.maximum(0.0, np.cos(2 * self.theta))
        self.c = self.tol.gaps(self.theta.size, self.seed_offset)
        self.vtilde = sliding_speed_base(self.theta, self.par)
        self.N = abs(self.par.z_f / (self.par.z_c - self.par.z_f))
        self.ca, self.sa = np.cos(self.par.alpha), np.sin(self.par.alpha)
        ring = RingParams(R=self.par.R, t=self.par.t, E=self.par.E,
                          nu=self.par.nu, w0=self.par.w0)
        self.FN_oval = normal_force_two_sided_cos2(ring) * self.b_face
        self.G = self.par.E / (2.0 * (1.0 + self.par.nu))

    # ------------------------------------------------------------------
    # Internal force negotiation
    # ------------------------------------------------------------------

    def _forces(self, dpsi: float, c: np.ndarray):
        a = self.par.R * self.ca * dpsi
        k = self.chain.k0_eff * self.g
        f_lead = k * np.maximum(0.0, a - c)
        f_trail = k * np.maximum(0.0, -a - c)
        return f_lead, f_trail

    def _torque_of(self, dpsi: float, c: np.ndarray, mu_wind: float = 0.0):
        f_lead, f_trail = self._forces(dpsi, c)
        fac = self.ca + mu_wind * self.sa
        return self.par.R * fac * float(f_lead.sum() - f_trail.sum())

    def _negotiate(self, T_out: float, c: np.ndarray | None = None,
                   mu_wind: float = 0.0) -> dict:
        c = self.c if c is None else c
        if T_out <= 0:
            f_lead, f_trail = self._forces(0.0, c)
            return dict(dpsi=0.0, f=f_lead, f_trail=f_trail)
        hi = 1e-3
        while self._torque_of(hi, c, mu_wind) < T_out:
            hi *= 2.0
        dpsi = brentq(lambda d: self._torque_of(d, c, mu_wind) - T_out, 0.0, hi)
        f_lead, f_trail = self._forces(dpsi, c)
        return dict(dpsi=dpsi, f=f_lead, f_trail=f_trail)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        T_out: float,
        omega_in: float,
        T: float | None = None,
    ) -> dict:
        """Compute a steady-state operating point.

        Parameters
        ----------
        T_out : float
            Output torque [N·mm].  Must be ≥ 0.
        omega_in : float
            WG angular speed [rad/s].  Use 0 for static / breakaway.
        T : float or None
            Drive temperature [°C].  ``None`` uses ``thermal.T_ref``.

        Returns
        -------
        dict
            Complete operating-point state.  Key entries:

            ``T_in`` : float
                Required input torque [N·mm].
            ``eta`` : float
                Mechanical efficiency ``T_out / (N × T_in)`` [-].
            ``T_ideal_in`` : float
                Ideal (lossless) input torque ``T_out / N`` [N·mm].
            ``T_teeth_in`` : float
                Tooth-flank friction loss, input-referred [N·mm].
            ``T_wg_in`` : float
                WG bearing drag, input-referred [N·mm].
            ``T_churn`` : float
                Churn + seal drag, input-referred [N·mm].
            ``dpsi_mesh`` : float
                Elastic mesh wind-up [rad].
            ``dpsi_total`` : float
                Total drive wind-up including structural chain [rad].
            ``K_mesh`` : float
                Instantaneous mesh torsional stiffness [N·mm/rad].
            ``K_total`` : float
                Total chain torsional stiffness [N·mm/rad].
            ``n_loaded`` : int
                Number of teeth carrying load.
            ``f_peak`` : float
                Maximum single-tooth force [N].
            ``f``, ``f_trail`` : numpy.ndarray
                Per-tooth leading / trailing flank forces [N].
            ``theta`` : numpy.ndarray
                Engaged-tooth angular positions [rad].
            ``mu_i`` : numpy.ndarray
                Per-tooth friction coefficient [-].
            ``v_i`` : numpy.ndarray
                Per-tooth sliding speed [mm/s].

        Examples
        --------
        >>> tw = HarmonicDriveTwin()
        >>> s = tw.solve(T_out=30_000, omega_in=209.4, T=20.0)
        >>> round(s['eta'] * 100, 1)
        75.8
        """
        T = self.thermal.T_ref if T is None else T
        vsc = self.thermal.visc_scale(T)
        nu_T = self.thermal.nu_ref * vsc
        c_eff = self.c + self.thermal.dc_flank(self.par.R, self.par.alpha)
        FN_oval = self.FN_oval * max(0.0, 1.0 + self.thermal.dw0(self.par.R)
                                     / self.par.w0)

        r = self._negotiate(T_out, c_eff)
        f, f_trail = r["f"], r["f_trail"]
        f_all = f + f_trail
        loaded = f > 1e-9

        v_i = self.vtilde * abs(omega_in)
        mu_i = (self.fric.mu(v_i, visc_scale=vsc)
                if omega_in != 0.0 else np.full_like(v_i, self.fric.mu_s))
        T_teeth_in = float(np.sum(mu_i * f_all * self.vtilde))
        P_teeth = T_teeth_in * abs(omega_in)

        F_cam = FN_oval + float(f_all.sum()) * self.sa
        v_surf = self.par.R * abs(omega_in)
        mu_wg = (self.fric.mu_wg_eff(v_surf) if omega_in != 0
                 else self.fric.mu_wg * self.fric.wg_breakaway)
        T_wg_in = mu_wg * self.par.R * F_cam
        P_wg = T_wg_in * abs(omega_in)

        T_churn = self.fric.T_drag(omega_in, nu_T)
        P_churn = T_churn * abs(omega_in)

        T_ideal_in = T_out / self.N
        T_in = T_ideal_in + T_teeth_in + T_wg_in + T_churn
        eta = (T_out / (self.N * T_in)) if T_in > 0 else 0.0

        K_mesh = (T_out / r["dpsi"]) if r["dpsi"] > 0 else np.inf
        K_tot, Ks = self.chain.series(K_mesh)
        dpsi_total = ((T_out / K_tot)
                      if np.isfinite(K_tot) and T_out > 0 else r["dpsi"])

        return dict(T_out=T_out, omega_in=omega_in, T_degC=T, nu_cSt=nu_T,
                    T_in=T_in, eta=eta,
                    T_ideal_in=T_ideal_in, T_teeth_in=T_teeth_in,
                    T_wg_in=T_wg_in, T_churn=T_churn,
                    P_teeth=P_teeth, P_wg=P_wg, P_churn=P_churn,
                    P_out=T_out * abs(omega_in) / self.N,
                    dpsi_mesh=r["dpsi"], dpsi_total=dpsi_total,
                    K_mesh=K_mesh, K_total=K_tot, K_chain=Ks,
                    n_loaded=int(loaded.sum()),
                    n_preloaded=int((f_trail > 1e-9).sum()),
                    f=f, f_trail=f_trail, f_peak=float(f.max(initial=0.0)),
                    theta=self.theta, mu_i=mu_i, v_i=v_i, F_cam=F_cam,
                    c_eff=c_eff)

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def hysteresis(
        self,
        T_max: float,
        n: int = 121,
        T: float | None = None,
    ) -> dict:
        """Quasi-static loading / unloading hysteresis loop.

        Parameters
        ----------
        T_max : float
            Maximum output torque [N·mm].
        n : int
            Number of points per branch.
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        dict
            Keys: ``T`` (torque array), ``dpsi_load``, ``dpsi_unload``,
            ``loop_width_rel``, ``dpsi_band_mid``, ``T_backdrive``,
            ``lag_zero_torque``, ``deadzone``.
        """
        mu = self.fric.mu_s
        T_c = self.thermal.T_ref if T is None else T
        c_eff = self.c + self.thermal.dc_flank(self.par.R, self.par.alpha)
        Ts = np.linspace(0.0, T_max, n)
        out = {}
        for lab, s in (("load", +mu), ("unload", -mu)):
            dps = []
            for Tq in Ts:
                r = self._negotiate(Tq, c_eff, mu_wind=s)
                K_mesh = (Tq / r["dpsi"]) if r["dpsi"] > 0 else np.inf
                extra = 0.0
                if Tq > 0:
                    K_tot, _ = self.chain.series(K_mesh)
                    extra = Tq / K_tot - r["dpsi"]
                dps.append(r["dpsi"] + extra)
            out[lab] = np.array(dps)
        width_T = 2.0 * mu * self.sa / self.ca
        i_mid = len(Ts) // 2
        dpsi_band = float(out["unload"][i_mid] - out["load"][i_mid])
        T_bd = self.backdrive_threshold(T=T_c)
        K_sec = Ts[-1] / out["load"][-1] if out["load"][-1] > 0 else np.inf
        lag_zero = T_bd / K_sec if np.isfinite(K_sec) else 0.0
        return dict(T=Ts, dpsi_load=out["load"], dpsi_unload=out["unload"],
                    loop_width_rel=width_T, dpsi_band_mid=dpsi_band,
                    T_backdrive=T_bd, lag_zero_torque=lag_zero,
                    deadzone=0.5 * self.tol.backlash / (self.par.R * self.ca))

    def hysteresis_dahl(
        self,
        T_max: float,
        n_per_branch: int = 200,
        psi_pre: float = 5.0e-5,
    ) -> dict:
        """Dahl presliding hysteresis loop.

        Parameters
        ----------
        T_max : float
            Peak output torque [N·mm].
        n_per_branch : int
            Points per half-cycle.
        psi_pre : float
            Dahl presliding length scale [rad].

        Returns
        -------
        dict
            Keys: ``psi``, ``T``, ``T_fric``, ``T_hold``, ``K``.
        """
        T_hold = self.backdrive_threshold()
        K = self.solve(T_max, 0.0)["K_total"]
        psi_max = (T_max + T_hold) / K
        seq = np.r_[np.linspace(0, psi_max, n_per_branch),
                    np.linspace(psi_max, -psi_max, 2 * n_per_branch),
                    np.linspace(-psi_max, psi_max, 2 * n_per_branch)]
        Tf = 0.0
        Tfs = []
        for i in range(len(seq)):
            if i > 0:
                dpsi = seq[i] - seq[i - 1]
                Tf += (T_hold / psi_pre) * (1 - np.sign(dpsi) * Tf / T_hold) * dpsi
                Tf = float(np.clip(Tf, -T_hold, T_hold))
            Tfs.append(Tf)
        return dict(psi=seq, T=K * seq + np.asarray(Tfs),
                    T_fric=np.asarray(Tfs), T_hold=T_hold, K=K)

    # ------------------------------------------------------------------
    # Sweeps and maps
    # ------------------------------------------------------------------

    def efficiency_curve(
        self,
        T_range: "np.ndarray",
        omega_in: float,
        T: float | None = None,
    ) -> dict:
        """Efficiency vs output torque at fixed speed and temperature.

        Parameters
        ----------
        T_range : array_like
            Output torque values [N·mm].
        omega_in : float
            WG angular speed [rad/s].
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        dict
            Keys: ``T`` (torque), ``eta``, ``T_in``, ``T_teeth``, ``T_wg``.
        """
        pts = [self.solve(Tq, omega_in, T) for Tq in T_range]
        return dict(T=np.asarray(T_range),
                    eta=np.array([p["eta"] for p in pts]),
                    T_in=np.array([p["T_in"] for p in pts]),
                    T_teeth=np.array([p["T_teeth_in"] for p in pts]),
                    T_wg=np.array([p["T_wg_in"] for p in pts]))

    def efficiency_map(
        self,
        T_range: "np.ndarray",
        omega_range: "np.ndarray",
        T: float | None = None,
    ) -> "np.ndarray":
        """2-D efficiency map over torque × speed.

        Parameters
        ----------
        T_range : array_like
            Output torque values [N·mm].
        omega_range : array_like
            WG angular speed values [rad/s].
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        numpy.ndarray of shape (len(omega_range), len(T_range))
            Efficiency values [-].
        """
        return np.array([[self.solve(Tq, om, T)["eta"] for Tq in T_range]
                         for om in omega_range])

    def breakaway_input_torque(self, T: float | None = None) -> float:
        """No-load starting torque at the WG input shaft [N·mm].

        Parameters
        ----------
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        float
            Starting input torque [N·mm].
        """
        return self.solve(0.0, 0.0, T)["T_in"]

    def backdrive_threshold(self, T: float | None = None) -> float:
        """Minimum output torque required to back-drive the gear [N·mm].

        Returns an upper bound (gross-slip mu_s).  The true value lies between
        ``N × breakaway_input_torque()`` (lower) and this result.

        Parameters
        ----------
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        float
            Back-drive threshold output torque [N·mm].
        """
        def resid(Tq):
            s = self.solve(Tq, 0.0, T)
            return 2.0 * Tq / self.N - s["T_in"]
        hi = 1e3
        while resid(hi) < 0:
            hi *= 2.0
            if hi > 1e9:
                return np.inf
        return brentq(resid, 0.0, hi)

    def monte_carlo(
        self,
        T_out: float,
        omega_in: float,
        n_mc: int = 64,
        T: float | None = None,
    ) -> dict:
        """Monte-Carlo sweep over tooth-clearance realisations.

        Parameters
        ----------
        T_out : float
            Output torque [N·mm].
        omega_in : float
            WG angular speed [rad/s].
        n_mc : int
            Number of independent clearance draws.
        T : float or None
            Drive temperature [°C].

        Returns
        -------
        dict
            Arrays ``eta``, ``dpsi``, ``n_loaded``, ``f_peak``, each of
            length *n_mc*.
        """
        etas, dps, nl, fpk = [], [], [], []
        for k in range(n_mc):
            tw = replace(self, seed_offset=k + 1)
            s = tw.solve(T_out, omega_in, T)
            etas.append(s["eta"]); dps.append(s["dpsi_total"])
            nl.append(s["n_loaded"]); fpk.append(s["f_peak"])
        arr = lambda x: np.asarray(x, dtype=float)  # noqa: E731
        return dict(eta=arr(etas), dpsi=arr(dps),
                    n_loaded=arr(nl), f_peak=arr(fpk))

    def warmup(
        self,
        T_out: float,
        omega_in: float,
        t_end: float = 3600.0,
        dt: float = 5.0,
        T_amb: float = 20.0,
        R_th: float = 1.5,
        C_th: float = 150.0,
    ) -> dict:
        """Thermal warm-up transient to steady state.

        Integrates ``C_th dT/dt = P_loss(T) - (T - T_amb) / R_th`` until
        thermal equilibrium.  Also estimates grease life at the steady
        temperature.

        Parameters
        ----------
        T_out : float
            Output torque [N·mm].
        omega_in : float
            WG angular speed [rad/s].
        t_end : float
            Simulation duration [s].
        dt : float
            Time step [s].
        T_amb : float
            Ambient temperature [°C].
        R_th : float
            Drive-to-ambient thermal resistance [K/W].
        C_th : float
            Drive thermal mass [J/K].

        Returns
        -------
        dict
            Keys: ``t``, ``T`` (temperature history), ``eta``, ``T_ss``,
            ``eta_ss``, ``P_loss_ss`` [W], ``L_grease_revs``,
            ``L_grease_h``.
        """
        ts = np.arange(0.0, t_end + dt, dt)
        Tarr = np.empty_like(ts); Tarr[0] = T_amb
        eta = np.empty_like(ts)
        for i in range(len(ts)):
            s = self.solve(T_out, omega_in, T=float(Tarr[i]))
            P_loss = (s["P_teeth"] + s["P_wg"] + s["P_churn"]) / 1000.0
            eta[i] = s["eta"]
            if i + 1 < len(ts):
                Tarr[i + 1] = Tarr[i] + (P_loss - (Tarr[i] - T_amb) / R_th) / C_th * dt

        def bal(Tq):
            s = self.solve(T_out, omega_in, T=Tq)
            return (s["P_teeth"] + s["P_wg"] + s["P_churn"]) / 1000.0 \
                - (Tq - T_amb) / R_th
        lo, hi = T_amb, T_amb + 150.0
        T_ss = brentq(bal, lo, hi) if bal(lo) * bal(hi) < 0 else Tarr[-1]
        s_ss = self.solve(T_out, omega_in, T=T_ss)
        L_grease = 6.0e9 * np.exp(-0.046 * T_ss)
        return dict(t=ts, T=Tarr, eta=eta, T_ss=T_ss, eta_ss=s_ss["eta"],
                    P_loss_ss=(s_ss["P_teeth"] + s_ss["P_wg"] + s_ss["P_churn"]) / 1000.0,
                    L_grease_revs=L_grease,
                    L_grease_h=L_grease / (abs(omega_in) / self.N / (2 * np.pi)) / 3600.0
                    if omega_in else np.inf)

    def wear_life(
        self,
        T_out: float,
        omega_in: float,
        revs_end: float = 1.5e9,
        n_step: int = 30,
        k_w: float = 2.0e-14,
    ) -> dict:
        """Archard flank-wear simulation over drive lifetime.

        Parameters
        ----------
        T_out : float
            Output torque [N·mm].
        omega_in : float
            WG angular speed [rad/s].
        revs_end : float
            Total WG revolutions to simulate.
        n_step : int
            Number of equally-spaced evaluation points.
        k_w : float
            Archard wear coefficient [mm/(N·mm·rev)] calibrated so lost-motion
            growth stays in the arcmin class over catalogue life.

        Returns
        -------
        dict
            Arrays ``revs``, ``lost`` [rad], ``f_peak`` [N],
            ``n_loaded``, ``wear_mean`` [mm].
        """
        kap = self.par.z_c / self.par.z_f
        thq = np.linspace(-np.pi / 4, np.pi / 4, 181)
        s_rev = 2.0 * np.trapezoid(sliding_speed_base(thq, self.par), thq) / kap
        c0 = self.c.copy()
        revs = np.linspace(0.0, revs_end, n_step + 1)
        drv = revs[1] - revs[0]
        out = dict(revs=revs, lost=np.zeros(n_step + 1),
                   f_peak=np.zeros(n_step + 1), n_loaded=np.zeros(n_step + 1),
                   wear_mean=np.zeros(n_step + 1))
        c_backup = self.c.copy()
        for j in range(n_step + 1):
            s = self.solve(T_out, omega_in)
            out["lost"][j] = self.solve(0.04 * T_out, omega_in)["dpsi_total"]
            out["f_peak"][j] = s["f_peak"]
            out["n_loaded"][j] = s["n_loaded"]
            out["wear_mean"][j] = float(np.mean(self.c - c0))
            if j == n_step:
                break
            f = s["f"] + s["f_trail"]
            hz_pk = hertz_flank(max(out["f_peak"][j], 1e-6), self.b_face)
            p_mean = 0.785 * hz_pk["p_max"] * np.sqrt(
                np.maximum(f, 0.0) / max(out["f_peak"][j], 1e-9))
            dh = k_w * p_mean * s_rev * drv
            self.c = self.c + dh
        self.c = c_backup
        return out

    def hertz_flank_summary(
        self,
        T_out: float = BASELINE.T_out,
        omega_in: float = 209.4,
    ) -> dict:
        """Hertz and EHL contact state at the most-loaded tooth.

        Parameters
        ----------
        T_out : float
            Output torque [N·mm].
        omega_in : float
            WG angular speed [rad/s].

        Returns
        -------
        dict
            Combined output of :func:`hertz_flank` and :func:`ehl_lambda`.
        """
        s = self.solve(T_out, omega_in)
        hz = hertz_flank(s["f_peak"], self.b_face)
        i_pk = int(np.argmax(s["f"]))
        lam = ehl_lambda(s["v_i"][i_pk], nu_cSt=s["nu_cSt"], Rx=hz["Rx"],
                         w_line=s["f_peak"] / self.b_face,
                         sigma_rough=self.fric.sigma_rough)
        return dict(**hz, **lam)
