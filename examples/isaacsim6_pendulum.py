"""Isaac Sim 6 integration example — harmonic-drive actuator twin.

Demonstrates how to use harmonic_drive_twin as a physics-based actuator model
inside NVIDIA Isaac Sim 6.  The simulation runs a 1-DOF pendulum (2 kg arm,
1 m length) driven by a Schaeffler RT1-H-17-100-CS harmonic-drive actuator.

Physical scenario
-----------------
* The arm starts at +45° from the vertical (hanging-down rest position).
* A PD position controller commands the arm back to 0°.
* At each physics step the commanded output torque is fed through the
  harmonic-drive twin, which returns the efficiency-corrected motor torque
  and the per-channel loss breakdown.
* Results are written to ``results/pendulum_test.csv``.

Prerequisites
-------------
    # Install the twin (NumPy + SciPy only)
    pip install harmonic-drive-twin

    # Install into the Isaac Sim 6 Python env (one-time)
    isaacsim6-python -m pip install harmonic-drive-twin

Usage
-----
    # Headless (fastest)
    isaacsim6-python examples/isaacsim6_pendulum.py

    # With GUI viewer
    isaacsim6-python examples/isaacsim6_pendulum.py --gui

    # Different gearbox size or step count
    isaacsim6-python examples/isaacsim6_pendulum.py --size 20 --steps 300

Notes
-----
* Run on ARM64 (NVIDIA GB10/DGX Spark): the LD_PRELOAD for libgomp is set
  automatically.  On x86_64 you can remove that line.
* The pendulum USD scene is built programmatically; no external .usda file
  is needed.
* Isaac Sim API used: World, Articulation, add_reference_to_stage.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

# ARM64 (Grace Blackwell / DGX Spark) requires libgomp pre-load.
# Remove this line on x86_64 systems.
os.environ.setdefault("LD_PRELOAD", "/lib/aarch64-linux-gnu/libgomp.so.1")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gui", action="store_true",
                    help="Launch Isaac Sim with the GUI viewport")
parser.add_argument("--steps", type=int, default=600,
                    help="Simulation steps (default 600 = 10 s at 60 Hz)")
parser.add_argument("--size", type=int, default=17,
                    choices=[14, 17, 20, 25],
                    help="TPI 275 catalogue size (default 17)")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Isaac Sim startup  (must come before any omni.* imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": not args.gui, "anti_aliasing": 0})

from isaacsim.core.api import World                           # noqa: E402
from isaacsim.core.prims import Articulation                  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf                 # noqa: E402
import omni.usd                                               # noqa: E402

# ---------------------------------------------------------------------------
# harmonic_drive_twin import
# ---------------------------------------------------------------------------
from harmonic_drive_twin import HarmonicDriveTwin, twin_for_size  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "pendulum_test.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 1.0 / 60.0        # physics timestep [s]
N_STEPS = args.steps
SIZE = args.size
RATIO = 100             # gear reduction ratio
T_DEGC = 20.0           # operating temperature [°C]

# PD controller gains (output-shaft side)
KP = 50.0               # N·m/rad
KD = 5.0                # N·m·s/rad
Q_TARGET = 0.0          # target joint angle [rad] — hanging down

# ---------------------------------------------------------------------------
# Build USD scene programmatically
# ---------------------------------------------------------------------------
# Isaac Sim 6 (Newton) articulation structure:
#   /World/Pendulum  ← ArticulationRootAPI
#     /Anchor        ← RigidBodyAPI (dynamic, fixed to world via FixedBase)
#     /FixedBase     ← PhysicsFixedJoint (body0=Anchor, no body1 → world)
#     /Arm           ← RigidBodyAPI + CollisionAPI (the moving link)
#     /HarmonicJoint ← PhysicsRevoluteJoint + DriveAPI:angular

def build_pendulum_scene(stage: Usd.Stage) -> None:
    """Populate the active USD stage with a 1-DOF pendulum articulation."""
    # Physics scene
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Articulation root xform
    pend = UsdGeom.Xform.Define(stage, "/World/Pendulum")
    UsdPhysics.ArticulationRootAPI.Apply(pend.GetPrim())

    # Tiny dynamic anchor body at pivot height z=1 m
    anchor = UsdGeom.Sphere.Define(stage, "/World/Pendulum/Anchor")
    anchor.CreateRadiusAttr(0.02)
    UsdGeom.XformCommonAPI(anchor).SetTranslate(Gf.Vec3d(0, 0, 1))
    UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(anchor.GetPrim())
    mass.CreateMassAttr(0.001)

    # Fix the anchor to world (body1 omitted → world frame)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Pendulum/FixedBase")
    fixed.CreateBody0Rel().SetTargets(["/World/Pendulum/Anchor"])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 1))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # Moving arm: capsule, 2 kg, CoM 0.5 m below pivot
    arm = UsdGeom.Capsule.Define(stage, "/World/Pendulum/Arm")
    arm.CreateHeightAttr(0.9)
    arm.CreateRadiusAttr(0.03)
    arm.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(arm).SetTranslate(Gf.Vec3d(0, 0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(arm.GetPrim())
    arm_mass = UsdPhysics.MassAPI.Apply(arm.GetPrim())
    arm_mass.CreateMassAttr(2.0)
    arm_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, -0.5))
    UsdPhysics.CollisionAPI.Apply(arm.GetPrim())

    # Revolute joint: Anchor → Arm, axis = Y
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Pendulum/HarmonicJoint")
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets(["/World/Pendulum/Anchor"])
    joint.CreateBody1Rel().SetTargets(["/World/Pendulum/Arm"])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.5))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)

    # Drive: force control (stiffness=0, tiny damping for numerical stability)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.05)
    drive.CreateMaxForceAttr(500.0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
print(f"\n[hd-sim] Isaac Sim 6 — harmonic-drive pendulum test")
print(f"[hd-sim] Gearbox: TPI 275, size {SIZE}, ratio {RATIO}:1, T={T_DEGC}°C")
print(f"[hd-sim] Steps: {N_STEPS}  dt: {DT*1e3:.1f} ms  total: {N_STEPS*DT:.1f} s\n")

world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
build_pendulum_scene(omni.usd.get_context().get_stage())
world.reset()

art = Articulation("/World/Pendulum")
art.initialize()
joint_idx = art.get_dof_index("HarmonicJoint")
if joint_idx < 0:
    print("[hd-sim] ERROR: HarmonicJoint DOF not found.")
    simulation_app.close()
    sys.exit(1)

# Set initial angle: arm at +45° from rest (hanging down)
q0 = math.radians(45.0)
pos0 = np.zeros((1, art.num_dof), dtype=np.float64)
pos0[0, joint_idx] = q0
art.set_joint_positions(pos0)

# Instantiate the harmonic-drive twin for the chosen catalogue size
twin = twin_for_size(SIZE)
print(f"[hd-sim] Twin: T_adh={twin.fric.T_adh:.1f} N·mm  "
      f"f0={twin.fric.f0_churn:.2f}  μk={twin.fric.mu_k:.3f}")

# CSV logging
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=[
    "step", "t_s", "theta_deg", "omega_out_rpm", "omega_in_rpm",
    "T_pd_Nm", "T_in_Nm", "eta_pct", "T_churn_Nmm", "T_teeth_Nmm", "T_wg_Nmm"])
writer.writeheader()

# Run simulation
world.play()
print(f"\n{'Step':>5}  {'θ°':>7}  {'ω_out rpm':>10}  {'T_pd N·m':>10}  "
      f"{'T_in N·m':>9}  {'η %':>6}  {'T_churn':>8}")
print("-" * 72)

for step in range(N_STEPS):
    t = step * DT

    q   = float(art.get_joint_positions(joint_indices=[joint_idx])[0, 0])
    dq  = float(art.get_joint_velocities(joint_indices=[joint_idx])[0, 0])

    # PD controller → commanded output torque
    T_pd = float(np.clip(KP * (Q_TARGET - q) - KD * dq, -200.0, 200.0))

    # harmonic-drive twin: compute motor-side torque and losses
    omega_out_rpm = dq * 60.0 / (2.0 * math.pi)
    omega_in_rpm  = float(np.clip(omega_out_rpm * RATIO,
                                  -twin.params.omega_in_max * 60 / (2*math.pi),
                                   twin.params.omega_in_max * 60 / (2*math.pi)))
    omega_in_rads = abs(omega_in_rpm) * 2.0 * math.pi / 60.0
    T_out_Nmm = abs(T_pd) * 1e3

    s = twin.solve(T_out_Nmm, omega_in_rads, T=T_DEGC)
    eta = s["eta"] if T_out_Nmm > 1.0 else 0.0
    T_in_Nm = s["T_in"] * 1e-3

    # Apply output torque directly to the joint
    efforts = np.zeros((1, art.num_dof), dtype=np.float32)
    efforts[0, joint_idx] = float(T_pd)
    art.set_joint_efforts(efforts)

    # Log
    writer.writerow(dict(
        step=step, t_s=round(t, 4),
        theta_deg=round(math.degrees(q), 3),
        omega_out_rpm=round(omega_out_rpm, 3),
        omega_in_rpm=round(omega_in_rpm, 1),
        T_pd_Nm=round(T_pd, 4),
        T_in_Nm=round(T_in_Nm, 4),
        eta_pct=round(eta * 100, 2),
        T_churn_Nmm=round(s.get("T_churn", float("nan")), 3),
        T_teeth_Nmm=round(s.get("T_teeth_in", float("nan")), 3),
        T_wg_Nmm=round(s.get("T_wg_in", float("nan")), 3),
    ))

    if step % 60 == 0 or step == N_STEPS - 1:
        print(f"{step:>5}  {math.degrees(q):>7.2f}  {omega_out_rpm:>10.2f}  "
              f"{T_pd:>10.3f}  {T_in_Nm:>9.4f}  {eta*100:>6.1f}  "
              f"{s.get('T_churn', 0):>8.2f}")

    world.step(render=args.gui)

world.stop()
csv_file.close()

print(f"\n[hd-sim] Done. CSV written to: {CSV_PATH}")
simulation_app.close()
