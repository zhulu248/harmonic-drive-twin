# harmonic-drive-twin

Physics-based digital twin for strain-wave (harmonic) gearboxes.

Validated against the **Schaeffler TPI 275** product catalogue
(series RT1-H-CS, ratio i = 100) across four frame sizes (14, 17, 20, 25)
and five independent catalogue quantities — with only **three global calibration
dials** and zero per-size fitting.

| Catalogue quantity | Validation approach |
|---|---|
| Efficiency at rated torque | Within ±3 % band (catalogue scatter) |
| No-load running torque | LSQ fit on f₀ (Palmgren churn) |
| Starting torque | LSQ fit on T_adh (grease adhesion) |
| Back-driving torque | Catalogue value falls in twin's presliding–gross-slip band |
| Torsional stiffness | FE-anchored physical chain, no stiffness calibration |

## Installation

```bash
pip install -e ".[dev]"          # editable install with test dependencies
```

The package depends on **NumPy** and **SciPy** only.
Optional: `matplotlib` for the dashboard figure.

The package is fully self-contained — **no companion repositories required**.
All physics helpers are bundled in `harmonic_drive_twin/flex_common/`.

## Quick start

```python
from harmonic_drive_twin import solve_point, twin_for_size

# One-shot functional call
s = solve_point(T_out_Nm=30.0, speed_rpm=2000, T_degC=20.0)
print(f"η = {s['eta']*100:.1f} %")   # → η = 77.9 %

# Catalogue-preset twin for size 17
tw = twin_for_size(17)
s  = tw.solve(T_out=31_000, omega_in=209.4, T=20.0)
print(f"η = {s['eta']*100:.1f} %")   # → η = 75.8 %  (catalogue: 77 ± 3 %)
```

## API overview

### Convenience functions

| Function | Description |
|---|---|
| `solve_point(T_out_Nm, speed_rpm, T_degC, size)` | Single-point solve; no class needed |
| `twin_for_size(size)` | Pre-scaled twin for a TPI 275 catalogue size |
| `validate()` | Full catalogue validation; returns list of result dicts |
| `calibrate()` | Derive the three calibration dials from the catalogue table |

### `HarmonicDriveTwin` methods

| Method | Returns |
|---|---|
| `solve(T_out, omega_in, T)` | Full operating-point dict (η, T_in, losses, stiffness, …) |
| `efficiency_curve(T_range, omega_in, T)` | η vs torque at fixed speed |
| `efficiency_map(T_range, omega_range, T)` | 2-D η array |
| `hysteresis(T_max, n, T)` | Quasi-static loading/unloading loop |
| `hysteresis_dahl(T_max, …)` | Dahl presliding smooth hysteresis |
| `breakaway_input_torque(T)` | No-load starting torque [N·mm] |
| `backdrive_threshold(T)` | Minimum output torque to back-drive [N·mm] |
| `monte_carlo(T_out, omega_in, n_mc, T)` | Scatter over tooth-clearance realisations |
| `warmup(T_out, omega_in, …)` | Thermal warm-up transient + grease life |
| `wear_life(T_out, omega_in, …)` | Archard flank-wear simulation over lifetime |
| `hertz_flank_summary(T_out, omega_in)` | Hertz contact + EHL film at peak-load tooth |

### Parameter blocks

```python
from harmonic_drive_twin import (
    MeshParams,       # gear geometry
    FrictionParams,   # Stribeck + Palmgren + churn/seal
    ThermalParams,    # viscosity–temperature model
    ToleranceParams,  # backlash + pitch scatter
    StiffnessChain,   # series torsional stiffness chain
)
```

## `solve` return dict — key fields

| Key | Unit | Description |
|---|---|---|
| `eta` | — | Mechanical efficiency |
| `T_in` | N·mm | Input torque |
| `T_teeth_in` | N·mm | Tooth-flank friction loss (input-referred) |
| `T_wg_in` | N·mm | WG bearing drag (input-referred) |
| `T_churn` | N·mm | Churn + seal drag (input-referred) |
| `dpsi_mesh` | rad | Elastic mesh wind-up |
| `dpsi_total` | rad | Total drive wind-up |
| `K_total` | N·mm/rad | Total torsional stiffness |
| `n_loaded` | — | Number of engaged teeth carrying load |
| `f_peak` | N | Peak single-tooth force |
| `f`, `f_trail` | N | Per-tooth leading / trailing flank forces |

## Running the tests

```bash
pytest tests/ -v
```

All tests pass in < 60 s on a modern workstation.

## Physics model summary

Three independent loss channels, each with a separate physical model:

1. **Tooth-flank friction** — Stribeck curve; film ratio λ ≈ 0.1 (boundary /
   mixed) confirms constant-μ is the correct regime.  Closed-form:
   T_flank = μ · T_out · J / (R cos α), J = 0.322 mm.

2. **WG bearing drag** — Palmgren M₁ = f₁ P dₘ (load-dependent, rolling-element
   bearing); f₁ ≈ 5×10⁻⁴ for thin-section ball bearings.

3. **Churn + seal** — Palmgren M₀ = 10⁻⁷ f₀ (ν n)^{2/3} dₘ³ (speed-dependent,
   load-independent).  At rated load this channel carries ~60 % of total loss —
   the load-free channel sets the catalogue η band, not the teeth.

Validated loss split (size 17, rated, +20 °C, 2000 rpm):

| Channel | N·mm | Share |
|---|---|---|
| Tooth flanks | 28 | 37 % |
| WG bearing | 6 | 8 % |
| Churn + seal | 51 | 67 % |

## Calibration protocol (rev-E)

Three dials, fitted once in physics-preferred order:

1. `T_adh` — grease adhesion breakaway; calibrated on the starting-torque row.
2. `f0_churn` — Palmgren churn factor; LSQ on no-load running-torque row.  Must
   land in [2, 4] (thin-section four-point-contact bearing range).
3. `mu_k` — boundary flank friction; mean-error fit on the efficiency row.  Must
   stay in [0.05, 0.12].

All other parameters (`T_seal`, `mu_wg`, `mu_v`, …) are set from literature /
Palmgren values and never touched.

## Isaac Sim 6 integration

The package ships a ready-to-run actuator model for
[NVIDIA Isaac Sim 6](https://developer.nvidia.com/isaac/sim):

### Quick start (Isaac Sim 6)

```bash
# Install the twin into the Isaac Sim 6 Python env (one-time)
isaacsim6-python -m pip install harmonic-drive-twin

# Run the included pendulum example (headless, 10 s simulation)
isaacsim6-python examples/isaacsim6_pendulum.py

# With GUI viewport
isaacsim6-python examples/isaacsim6_pendulum.py --gui --size 20
```

### Embedding the actuator in your own robot

```python
# In your Isaac Sim 6 script (after SimulationApp starts)
from harmonic_drive_twin import twin_for_size
from isaacsim.core.prims import Articulation
import numpy as np, math

twin = twin_for_size(17)        # TPI 275 size 17, ratio 100:1
art  = Articulation("/World/MyRobot")
art.initialize()
joint_idx = art.get_dof_index("DriveJoint")

for step in range(N):
    q   = float(art.get_joint_positions(joint_indices=[joint_idx])[0, 0])
    dq  = float(art.get_joint_velocities(joint_indices=[joint_idx])[0, 0])

    # your torque command (N·m)
    T_cmd = my_controller(q, dq)

    # compute motor torque + losses via the twin
    s = twin.solve(abs(T_cmd) * 1e3,           # N·mm
                   abs(dq) * abs(RATIO),        # motor rad/s
                   T=20.0)
    print(f"η = {s['eta']*100:.1f}%  T_in = {s['T_in']*1e-3:.3f} N·m")

    # apply to joint
    efforts = np.zeros((1, art.num_dof), dtype=np.float32)
    efforts[0, joint_idx] = float(T_cmd)
    art.set_joint_efforts(efforts)
    world.step()
```

### Example output (pendulum test, size 17, 100:1, T=20°C)

| Time [s] | θ [°] | ω_out [rpm] | T_cmd [N·m] | η [%] | T_churn [N·mm] |
|---|---|---|---|---|---|
| 0.00 | 45.0 | 0.0 | −39.3 | 81.7 | 26.1 |
| 0.50 | 11.3 | −26.8 | −11.4 | 84.1 | 41.9 |
| 1.00 | −5.3 | −10.4 | 4.5 | 82.9 | 38.6 |
| 3.00 | 0.0 | 0.0 | 0.0 | 0.0 | 5.4 |

The arm settles from 45° to 0° in ~3 s with η ≈ 80–85% during motion.

### Isaac Sim 6 articulation notes

Isaac Sim 6 uses the **Newton** physics engine. USD articulations must follow
this structure for Newton to recognise the DOFs:

```
/World/MyRobot   ← PhysicsArticulationRootAPI
  /Base          ← RigidBodyAPI (dynamic, fixed to world via PhysicsFixedJoint)
  /FixedBase     ← PhysicsFixedJoint (body0=Base, no body1 → world-fixed)
  /Link1         ← RigidBodyAPI + CollisionAPI
  /Joint1        ← PhysicsRevoluteJoint + PhysicsDriveAPI:angular
```

Common pitfalls:
* **Do not** use `prim_path="/World"` in `add_reference_to_stage` — it
  conflicts with Isaac Sim's own `/World` stage prim.  Use `/World/MyRobot`.
* **Do not** make the anchor body `kinematicEnabled = true` — Newton does not
  support kinematic bodies inside articulations.  Use a dynamic body + a
  fixed joint instead.
* The joint DOF name is the USD prim name of the `PhysicsRevoluteJoint`.

## Project structure

```
harmonic_drive_twin/
├── harmonic_drive_twin/
│   ├── __init__.py          public API
│   ├── twin.py              HarmonicDriveTwin class
│   ├── params.py            parameter dataclasses
│   ├── catalogue.py         TPI 275 data + validation
│   ├── flex_common/         bundled physics helpers (self-contained)
│   └── py.typed
├── tests/
│   └── test_catalogue_validation.py
├── examples/
│   ├── basic_usage.py              standalone Python example
│   ├── isaacsim6_pendulum.py       Isaac Sim 6 pendulum demo
│   └── isaacsim6_actuator.py       HarmonicDriveActuator class
└── pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
