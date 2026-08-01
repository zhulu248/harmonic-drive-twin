# What's inside this package — a gentle tour

This document explains **what you are shipping** when you release
`harmonic-drive-twin`, piece by piece, in plain language.

It is written for two readers:

* **You**, deciding what belongs in the release.
* **A stranger** who has just cloned the repo and wants to know what
  they are looking at.

If you only read one paragraph, read this one:

> The package is a **physics calculator for a strain-wave gearbox**. You hand
> it an operating point — "I need 30 N·m out, the motor is spinning at
> 2000 rpm, it's 20 °C in there" — and it hands back the motor torque you
> actually need, the efficiency, where every watt of loss went, and how much
> the gearbox twists under that load. Everything else in the repo exists to
> support, prove, or demonstrate that one calculation.

---

## 1. The map

```
harmonic-drive-twin/
│
├── pyproject.toml          ← the shipping label (name, version, dependencies)
├── LICENSE                 ← MIT — permission to use it
├── README.md               ← the shop window (what it does, quick start, API)
├── .gitignore              ← what never leaves your machine
│
├── harmonic_drive_twin/    ← ★ THE PRODUCT — everything below is the model
│   ├── __init__.py             the front door
│   ├── twin.py                 the engine
│   ├── params.py               the dials
│   ├── catalogue.py            the reality check
│   ├── py.typed                a flag saying "type hints included"
│   └── flex_common/            the physics foundation (7 modules)
│       ├── baseline.py
│       ├── ring.py
│       ├── ellipse.py
│       ├── mesh.py
│       ├── profile.py
│       ├── friction.py
│       └── teeth.py
│
├── examples/               ← on-ramps: three runnable scripts
│   ├── basic_usage.py
│   ├── isaacsim6_actuator.py
│   └── isaacsim6_pendulum.py
│
├── tests/                  ← the proof it still works
│   └── test_catalogue_validation.py
│
└── docs/                   ← the manual
    └── user_guide/
        ├── user_guide.pdf      10-page branded guide
        ├── user_guide.tex      its source
        └── assets/             logos + the pendulum result figure
```

Total: **2 947 lines of Python** in the model itself (3 633 counting examples
and tests). Small. That is a feature — a reviewer can read the whole model in
an afternoon.

---

## 2. The four layers, from the bottom up

The package is built in layers. Each layer only knows about the ones below it.
This is why it stays understandable.

```
      ┌─────────────────────────────────────────────┐
  4   │  __init__.py    the front door              │  ← what users import
      ├─────────────────────────────────────────────┤
  3   │  twin.py        the engine                  │  ← does the work
      ├─────────────────────────────────────────────┤
  2   │  params.py      the dials                   │  ← what you can change
      ├─────────────────────────────────────────────┤
  1   │  flex_common/   the physics foundation      │  ← the equations
      └─────────────────────────────────────────────┘

      catalogue.py sits alongside 2–3 and asks: "does this match reality?"
```

---

### Layer 1 — `flex_common/` : the physics foundation

**~1 300 lines. Seven small modules. Each is one piece of textbook mechanics.**

These are the equations that describe *how a harmonic drive deforms and
meshes*. They were originally the shared library of your multi-topic FEA study
and were bundled into the package so it stands alone — a user does **not** need
your research repo to run the model.

| Module | Lines | In one sentence |
|---|---|---|
| `baseline.py` | 79 | **The reference part.** One agreed set of numbers — R = 20 mm, wall 0.4 mm, ovalization 0.30 mm, 200/202 teeth — so every other module is talking about the same gearbox. |
| `ring.py` | 170 | **The flexspline as a thin ring.** Squash a thin steel ring into an ellipse; this computes the resulting bending moment, shear, and contact pressure — in closed form. Handles both "ring tied to the cam" and "cam can only push outward". |
| `ellipse.py` | 111 | **A real ellipse is not a pure cos 2θ.** Decomposes the true elliptical shape into harmonics (cos 2θ, cos 4θ, …) and adds the ring's response to each. The correction term for `ring.py`. |
| `mesh.py` | 305 | **Which teeth are touching, right now.** Given the wave-generator angle, it says which flexspline teeth line up with which circular-spline spaces, and how each tooth moves: radial breathing, tangential slide, tilt. This is the heart of the kinematics. |
| `profile.py` | 433 | **What shape the teeth must be.** Synthesises the circular-spline flank as the conjugate envelope of the flexspline tooth swept through the mesh, and computes the sliding speed between flanks — which feeds friction and wear. |
| `friction.py` | 74 | **Why it takes torque to spin an unloaded drive.** Without friction the no-load torque is exactly zero; this module shows why, and what friction adds. |
| `teeth.py` | 184 | **Tooth-root stress.** Lewis cantilever bending plus Dolan-Broghamer fillet concentration — the fatigue surrogate. |

> **Gentle note:** a user of the twin never needs to open these. They are the
> foundation under the floorboards. But they are what makes the model
> *physics-based* rather than a curve fit — and a skeptical reviewer will go
> straight here.

---

### Layer 2 — `params.py` : the dials

**425 lines. Four dataclasses. This is "everything you are allowed to change".**

Instead of magic numbers scattered through the code, every adjustable quantity
lives in one of four labelled boxes:

| Dataclass | What it holds | Example knobs |
|---|---|---|
| `FrictionParams` | How things rub | boundary friction coefficient, Stribeck transition speed, wave-generator bearing coefficient, grease churn factor |
| `ThermalParams` | How temperature changes things | oil viscosity vs temperature, thermal expansion of clearances |
| `ToleranceParams` | Manufacturing imperfection | backlash, tooth-to-tooth pitch scatter (used for Monte-Carlo runs) |
| `StiffnessChain` | How the drive twists | the series chain: cup + wave-generator bearing + mesh + output bearing |

**Why this matters for release:** a user adapting the twin to *their* gearbox
edits parameters here — never the equations. That's the contract.

Units are stated once, at the top of the file, and hold everywhere:
**mm, N, N·mm, rad/s, °C.**

---

### Layer 3 — `twin.py` : the engine

**787 lines. One class, `HarmonicDriveTwin`, plus three contact helpers.**

This is the actual model. It takes the equations from Layer 1, the numbers from
Layer 2, and answers questions.

**The main method is `solve()`:**

```
                 you give it                    it gives you back
        ┌──────────────────────┐        ┌────────────────────────────────┐
        │  T_out   (torque)    │        │  T_in      motor torque needed │
        │  omega_in (speed)    │  ───▶  │  eta       efficiency          │
        │  T       (temp °C)   │        │  T_teeth_in  ┐                 │
        └──────────────────────┘        │  T_wg_in     ├ where the loss  │
                                        │  T_churn     ┘ actually went   │
                                        │  dpsi_total  how much it twists│
                                        │  K_total     torsional stiffness│
                                        └────────────────────────────────┘
```

The loss split into those **three channels** is the model's central idea:

1. **`T_teeth_in`** — tooth flanks rubbing. Boundary-lubricated, not oil-film.
2. **`T_wg_in`** — the wave-generator bearing rolling.
3. **`T_churn`** — grease churn + seal drag. Load-*independent*, and at rated
   torque it is the **biggest** of the three.

That third one is the non-obvious finding your study produced, and it is why
catalogue efficiency sits at 67–78 % rather than the ~90 % a
tooth-friction-only model predicts.

**Beyond `solve()`, the class also answers:**

| Method | Question it answers |
|---|---|
| `efficiency_curve()` | How does η vary with torque? |
| `efficiency_map()` | 2-D η map over torque × speed |
| `hysteresis()` / `hysteresis_dahl()` | What does the torque–windup loop look like? (lost motion) |
| `breakaway_input_torque()` | What torque starts it from cold rest? |
| `backdrive_threshold()` | Can the load push the motor backwards? |
| `monte_carlo()` | What's the scatter if tooth clearances vary? |
| `warmup()` | Thermal transient + grease life |
| `wear_life()` | Archard flank wear over a lifetime |
| `hertz_flank_summary()` | Contact pressure + oil film at the hardest-loaded tooth |

Plus three standalone helpers anyone can use: `hertz_flank()`, `ehl_lambda()`,
`wg_bearing_stiffness()`.

---

### Layer 4 — `__init__.py` : the front door

**129 lines. Almost all of it is documentation.**

This file does three jobs, and its job is *curation*:

1. **Decides what is public.** The `__all__` list is the promise: these 13 names
   are the supported API. Everything else is internal and may change.
2. **Provides `solve_point()`** — a one-line functional wrapper so a casual user
   never has to instantiate a class:
   ```python
   s = solve_point(T_out_Nm=30.0, speed_rpm=2000)   # → η = 77.9 %
   ```
3. **Carries the quick-start docstring**, written as doctests — so the examples
   in the docs are *executable* and cannot silently go stale.

---

## 3. The reality check — `catalogue.py` and `tests/`

This is what separates a digital twin from a plausible-looking spreadsheet.

### `catalogue.py` (250 lines)

Holds the **digitised Schaeffler TPI 275 table** (series RT1-H-CS, i = 100,
+20 °C) for four frame sizes — 14, 17, 20, 25:

```
size      14      17      20      25
R [mm]  17.78   21.59   25.40   31.75     pitch radius
T_N     10.0    31.0    52.0    87.0      rated torque [N·m]
eta     67.0    77.0    77.0    77.0      efficiency [%]
T_nlrt  35      51      105     195       no-load running torque [mN·m]
T_nlst  21      29      37      69        starting torque [mN·m]
T_bt    2.21    3.06    3.89    7.26      back-driving torque [N·m]
K1/K3   4.7k…   10k…    16k…    31k…      torsional stiffness [N·m/rad]
```

And three functions:

* **`twin_for_size(14|17|20|25)`** — a twin pre-scaled to that catalogue size.
* **`calibrate()`** — derives the **three global calibration dials** from the
  catalogue table.
* **`validate()`** — runs everything and reports how close the model lands.

> **The headline claim of the whole package:** five independent catalogue
> quantities, four sizes, matched with only **three global dials** and
> **zero per-size fitting**. Torsional stiffness uses no fitting at all — it
> comes from the FE-anchored physical chain.

### `tests/test_catalogue_validation.py` (183 lines)

Four tests that keep that claim honest:

| Test | Guards against |
|---|---|
| `test_calibration_reproducible()` | calibration drifting when code changes |
| `test_solve_point_size17()` | the size-17 number (75.8 % vs catalogue 77 ± 3 %) moving |
| `test_solve_point_default()` | the baseline number (77.9 %) moving |
| `test_twin_for_invalid_size()` | silently accepting a size that doesn't exist |

Run them with `pytest`. If they pass, the numbers in the README are true.

---

## 4. The on-ramps — `examples/`

Three scripts, in increasing order of ambition:

| Script | Lines | What it shows |
|---|---|---|
| `basic_usage.py` | 77 | **Start here.** Pure Python, no simulator. Single-point solve, catalogue-preset twin, full validation table printed to the terminal. |
| `isaacsim6_actuator.py` | 156 | The **reusable class** — `HarmonicDriveActuator`. Drop it into any Isaac Sim robot script: it reads joint state, computes the η-corrected motor torque, applies it, and logs the loss breakdown. |
| `isaacsim6_pendulum.py` | 270 | The **complete demo.** A 1-DOF pendulum (2 kg arm, 1 m) driven through a real RT1-H-17-100-CS. Builds the scene programmatically, runs headless or with GUI, writes CSV. Arm settles 45° → 0° in ~3 s at η 80–85 %. |

The pendulum example is the one that makes the model *credible to a roboticist*
— it proves the twin runs inside a real physics engine at real-time step rates,
not just in a notebook.

---

## 5. The manual — `docs/`

`user_guide.pdf` — 10 pages, Schaeffler-branded, built from `user_guide.tex`.
Its ten sections: what the model is → how the three loss channels work →
required inputs → outputs → Python quick start → Isaac Sim 6 integration →
the pendulum verification run → calibration parameters → running the tests →
troubleshooting.

This document you are reading now is its companion: the *guide* tells a user
**how to drive**; this one tells them **what's under the hood**.

---

## 6. How a user actually meets the package

There are three depths, and most people stop at the first:

```
DEPTH 1  ·  "just give me a number"          →  solve_point(30.0, 2000)
             one line, no classes                 ~30 seconds to first result

DEPTH 2  ·  "model my specific gearbox"      →  twin_for_size(17), or build a
             swap in your own parameters         HarmonicDriveTwin with your
                                                 own MeshParams/FrictionParams

DEPTH 3  ·  "put it in my robot sim"         →  copy isaacsim6_actuator.py,
             live actuator inside Isaac Sim      call .step() each frame
```

---

## 7. Before you release — three things to look at

Findings from the current repo state, not blockers, but worth a decision:

1. **Schaeffler logo files are tracked in a public MIT repo.**
   `docs/user_guide/assets/schaeffler_wordmark_green.png` and
   `we_pioneer_motion_claim.png` are corporate trademarks under an MIT license
   header. Consider shipping an unbranded PDF publicly and keeping the branded
   build internal.

2. **LaTeX build artefacts are tracked.**
   `user_guide.aux`, `.fls`, `.fdb_latexmk`, `.log`, `.out`, `.toc`, `.xdv`,
   and `build.log` add churn to every commit. Ship `user_guide.pdf` and
   `user_guide.tex`; gitignore the rest.

3. **Nine files are currently modified and uncommitted** — all in
   `docs/user_guide/`, including `user_guide.tex` and the PDF. Commit or revert
   before tagging a release.

Good news: `build/`, `*.egg-info/`, `.pytest_cache/`, and `__pycache__/` are
already correctly ignored, and the package tree itself is clean.

---

## 8. One-page summary

| Piece | Size | Role | Does a user open it? |
|---|---|---|---|
| `pyproject.toml` | 37 | shipping label | no |
| `README.md` | 249 | shop window | **yes, first** |
| `__init__.py` | 129 | front door, public API | indirectly |
| `twin.py` | 787 | the engine | if curious |
| `params.py` | 425 | the dials | **yes, to adapt** |
| `catalogue.py` | 250 | reality check | to validate |
| `flex_common/` × 7 | ~1 300 | physics foundation | rarely |
| `examples/` × 3 | 503 | on-ramps | **yes, to start** |
| `tests/` | 183 | proof | to verify |
| `docs/user_guide.pdf` | 10 pp | the manual | **yes** |

Dependencies: **NumPy and SciPy only.** Matplotlib optional, for plots.
Python ≥ 3.10. No compiled extensions, no companion repo, no license server.
That is why it installs anywhere in one command.
