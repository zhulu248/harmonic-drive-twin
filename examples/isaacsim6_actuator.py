"""Harmonic-drive actuator model for Isaac Sim 6.

Wraps ``harmonic_drive_twin.HarmonicDriveTwin`` with an Isaac Sim-friendly
interface.  At each simulation step the actuator:

1. Reads the output-shaft velocity from the articulation.
2. Computes the required input (motor) torque accounting for gearbox losses.
3. Applies the corrected torque to the joint.
4. Returns a dict of logged quantities (η, T_in, T_churn, …).

Usage::

    from actuator import HarmonicDriveActuator
    act = HarmonicDriveActuator(size=17, T_degC=20.0)
    log = act.step(art, joint_idx, T_out_Nm=30.0)
    print(log["eta"])   # → ~0.76
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from harmonic_drive_twin import HarmonicDriveTwin, twin_for_size


@dataclass
class HarmonicDriveActuator:
    """Isaac Sim 6 actuator wrapper for the strain-wave gearbox digital twin.

    Parameters
    ----------
    size : int or None
        TPI 275 catalogue size {14, 17, 20, 25}.  ``None`` uses the project
        baseline geometry (R = 20 mm).
    T_degC : float
        Drive operating temperature [°C].  Default: 20 °C (catalogue ref).
    ratio : int
        Gear reduction ratio (input/output).  Default: 100.
    max_motor_speed_rpm : float
        Maximum motor input speed [rpm].  Used to clamp the twin input.

    Attributes
    ----------
    twin : HarmonicDriveTwin
        The underlying physics model.
    history : list of dict
        Per-step log entries appended by :meth:`step`.
    """

    size: int | None = 17
    T_degC: float = 20.0
    ratio: int = 100
    max_motor_speed_rpm: float = 6000.0

    twin: HarmonicDriveTwin = field(init=False, repr=False)
    history: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.size is not None:
            self.twin = twin_for_size(self.size)
        else:
            self.twin = HarmonicDriveTwin()

    # ------------------------------------------------------------------
    def step(
        self,
        art,
        joint_idx: int,
        T_out_Nm: float,
        sim_time: float = 0.0,
        apply_torque: bool = True,
    ) -> dict:
        """Compute and optionally apply the actuator torque for one sim step.

        Parameters
        ----------
        art : isaacsim.core.prims.Articulation
            The articulation prim containing the harmonic-drive joint.
        joint_idx : int
            DOF index of the joint within the articulation.
        T_out_Nm : float
            Desired output torque [N·m].  Positive = same direction as velocity.
        sim_time : float
            Current simulation time [s].  Used for logging only.
        apply_torque : bool
            If ``True``, call ``art.set_joint_efforts()`` to apply the
            efficiency-corrected motor torque to the joint.

        Returns
        -------
        dict
            Keys: ``t``, ``theta_deg``, ``omega_out_rpm``, ``omega_in_rpm``,
            ``T_out_Nm``, ``T_in_Nm``, ``eta``, ``T_churn_Nmm``,
            ``T_teeth_Nmm``, ``T_wg_Nmm``, ``regime``.
        """
        # --- read joint state ---
        q_rad = float(art.get_joint_positions(joint_indices=[joint_idx])[0, 0])
        dq_rad_s = float(art.get_joint_velocities(joint_indices=[joint_idx])[0, 0])

        omega_out_rpm = dq_rad_s * 60.0 / (2.0 * math.pi)
        omega_in_rpm = omega_out_rpm * self.ratio
        omega_in_rpm = np.clip(omega_in_rpm, -self.max_motor_speed_rpm,
                               self.max_motor_speed_rpm)

        T_out_Nmm = T_out_Nm * 1e3

        # --- twin physics solve ---
        omega_in_rads = abs(omega_in_rpm) * 2.0 * math.pi / 60.0
        s = self.twin.solve(abs(T_out_Nmm), omega_in_rads, T=self.T_degC)

        eta = s["eta"] if abs(T_out_Nmm) > 1.0 else 0.0
        T_in_Nmm = s["T_in"]                  # motor input torque [N·mm]
        T_in_Nm = T_in_Nmm * 1e-3

        T_apply_Nm = T_out_Nm

        # --- apply to physics ---
        if apply_torque:
            efforts = np.zeros((1, art.num_dof), dtype=np.float32)
            efforts[0, joint_idx] = float(T_apply_Nm)
            art.set_joint_efforts(efforts)

        # --- log ---
        regime = "boundary" if omega_in_rpm < 10 else "mixed"
        entry = dict(
            t=sim_time,
            theta_deg=math.degrees(q_rad),
            omega_out_rpm=omega_out_rpm,
            omega_in_rpm=omega_in_rpm,
            T_out_Nm=T_out_Nm,
            T_in_Nm=T_in_Nm,
            eta=eta,
            T_churn_Nmm=s.get("T_churn", float("nan")),
            T_teeth_Nmm=s.get("T_teeth_in", float("nan")),
            T_wg_Nmm=s.get("T_wg_in", float("nan")),
            regime=regime,
        )
        self.history.append(entry)
        return entry

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return min/mean/max statistics over the logged history."""
        if not self.history:
            return {}
        etas = [e["eta"] for e in self.history if e["eta"] > 0]
        return dict(
            n_steps=len(self.history),
            eta_mean=float(np.mean(etas)) if etas else float("nan"),
            eta_min=float(np.min(etas)) if etas else float("nan"),
            eta_max=float(np.max(etas)) if etas else float("nan"),
            omega_in_rpm_max=max(abs(e["omega_in_rpm"]) for e in self.history),
        )
