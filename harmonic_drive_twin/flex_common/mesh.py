"""
Tooth-position / mesh-kinematics generator for the strain-wave gear (shared).

This is the Topic-4 (teeth engagement) core, promoted to ``common`` because the
tooth-placement and material-point (u, v, tilt) helpers are also needed by Topic-5
(profile design).  It answers Part (A) of the topic brief: *which* flexspline (FS)
teeth line up with circular-spline (CS) tooth spaces at any wave-generator (WG)
angle, and how the +/-45 deg wrap maps to actual teeth.

Frames & conventions (consistent with common.ring)
---------------------------------------------------
phi     : WG major-axis angle in the fixed (CS) frame                       [rad]
psi_f   : rigid-body rotation of the FS ring (its slow output rotation)     [rad]
theta   : angle measured FROM the current WG major axis, theta = beta - phi [rad]
          theta = 0 -> major axis (max outward push), theta = pi/2 -> minor axis
beta    : material angle of an FS point in the fixed frame (before deform.) [rad]
u = w   : radial displacement of the FS neutral ring, OUTWARD positive      [mm]
v       : tangential displacement of the FS neutral ring, +theta positive   [mm]
chi     : tooth (cross-section) tilt, right-handed about the axial dir      [rad]

Material-point motion of the inextensible neutral ring (leading 2nd harmonic)
-----------------------------------------------------------------------------
    u = w  =  w0 cos 2theta
    v      = -(w0/2) sin 2theta            (inextensibility  v' = -w)
    chi    = (w' - v)/R = -(3 w0 / 2R) sin 2theta
For the EXACT ellipse, u/v/chi are the same sums over harmonics n = 2,4,6,...:
    w   =  sum_n  w_n cos n theta
    v   = -sum_n (w_n / n) sin n theta
    chi = -sum_n  w_n (n^2 - 1)/n * sin n theta / R
(harmonics w_n from common.ellipse.ellipse_harmonics; n=2 recovers the above).

Mesh / contact reduction (seed for Part B force negotiation)
------------------------------------------------------------
* Radial wrap: an FS tooth is pushed into the CS ring where w(theta) > 0, i.e.
  |theta| < 45 deg -- EXACTLY the unilateral (bare-cam) wrap of common.ring.
  ~ z_f/4 teeth per lobe lie in this arc.
* Angular registration: FS pitch 2pi/z_f differs from CS pitch 2pi/z_c (z_c=z_f+2),
  so a tooth seated dead-centre in a CS space at the lobe drifts off-centre away
  from it.  The tangential offset of FS tooth k from its nearest CS space centre,
      t_k = e_k * r_k       (e_k = registration error [rad]),
  closes the flank backlash b_h and then penetrates the CS flank.  The penetration
  resolved along the line of action (pressure angle alpha) is the per-tooth spring
  compression handed to Part B:
      delta_k = max(0, |t_k| - b_h) * cos(alpha)      (loaded flank picked by sign).

Units: mm, N, rad (angles), MPa.  theta = 0 is the major axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys

import numpy as np

from harmonic_drive_twin.flex_common.ring import RingParams               # noqa: E402
from harmonic_drive_twin.flex_common.ellipse import ellipse_harmonics      # noqa: E402


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass
class MeshParams:
    """Geometry of the FS/CS tooth mesh.  Defaults = project baseline."""
    z_f: int = 200            # flexspline tooth count                         [-]
    z_c: int = 202            # circular-spline tooth count (= z_f + 2)         [-]
    R: float = 20.0           # FS pitch / mean-rim radius                     [mm]
    w0: float = 0.30          # WG radial ovalization at major axis            [mm]
    alpha_deg: float = 20.0   # pressure angle                                 [deg]
    t: float = 0.4            # rim wall thickness (for RingParams)            [mm]
    E: float = 2.0e5          # Young's modulus                                [MPa]
    nu: float = 0.30          # Poisson ratio                                  [-]
    # contact/clearance model
    backlash: float = 0.0     # circular flank backlash (total)               [mm]
    n_max: int = 10           # ellipse harmonics kept when exact=True         [-]

    @property
    def phase0(self) -> float:
        """Assembly phase [rad]: rigid FS offset that puts a well-registered
        beat node (FS tooth meshing a CS space) on the major axis at phi = 0.
        Derived from the tooth-count beat:  psi_f = pi/z_f - 2 phi/z_f."""
        return np.pi / self.z_f

    @property
    def m(self) -> float:
        """Gear module m = 2R / z_f [mm]  (FS and CS share the module)."""
        return 2.0 * self.R / self.z_f

    @property
    def R_c(self) -> float:
        """CS pitch radius R_c = m z_c / 2 = R + m [mm]."""
        return 0.5 * self.m * self.z_c

    @property
    def alpha(self) -> float:
        return np.deg2rad(self.alpha_deg)

    @property
    def b_h(self) -> float:
        """Half backlash (per flank) as an arc length [mm]."""
        return 0.5 * self.backlash

    def ring(self) -> RingParams:
        return RingParams(R=self.R, t=self.t, E=self.E, nu=self.nu, w0=self.w0)

    # harmonic content of the FS backbone -------------------------------------
    def harmonics(self) -> dict:
        """{n: w_n} even harmonics of the exact-ellipse neutral line."""
        h = ellipse_harmonics(self.R, self.w0, n_max=self.n_max)
        return {n: h["wn"][n] for n in h["ns"] if n >= 2}

    @property
    def kinematic_ratio(self) -> float:
        """d psi_f / d phi for CS fixed, FS output = -(z_c - z_f)/z_f.
        Magnitude 1/ratio; sign: FS creeps OPPOSITE to WG."""
        return -(self.z_c - self.z_f) / self.z_f

    def kinematic_psi_f(self, phi):
        """Rigid (tooth-counting) FS rotation that keeps the beat node on the
        major axis: psi_f = phase0 + kinematic_ratio * phi.  This is the NAIVE
        kinematic value Part B will contrast against the force-balanced one."""
        return self.phase0 + self.kinematic_ratio * np.asarray(phi, dtype=float)


# --------------------------------------------------------------------------- #
#  Material-point motion  (u, v, tilt)  -- PROMOTED, shared with Topic 5
# --------------------------------------------------------------------------- #
def point_motion(theta, par: MeshParams, exact: bool = False) -> dict:
    """Radial u=w, tangential v and tooth tilt chi of the inextensible FS
    neutral ring, as functions of angle-from-major-axis theta.

    exact=False -> pure 2nd harmonic (w0 cos2theta) closed form (default).
    exact=True  -> sum over ellipse harmonics n = 2,4,...,n_max.
    Returns w, v, tilt (all broadcast to theta's shape) plus dw/dtheta.
    """
    theta = np.asarray(theta, dtype=float)
    R, w0 = par.R, par.w0
    if not exact:
        c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
        w = w0 * c2
        wp = -2.0 * w0 * s2
        v = -(w0 / 2.0) * s2
        tilt = -(3.0 * w0 / (2.0 * R)) * s2
        return dict(theta=theta, w=w, wp=wp, v=v, tilt=tilt)

    w = np.zeros_like(theta); wp = np.zeros_like(theta)
    v = np.zeros_like(theta); tilt = np.zeros_like(theta)
    for n, wn in par.harmonics().items():
        cn, sn = np.cos(n * theta), np.sin(n * theta)
        w += wn * cn
        wp += -n * wn * sn
        v += -(wn / n) * sn
        tilt += -(wn * (n ** 2 - 1) / n) * sn / R
    return dict(theta=theta, w=w, wp=wp, v=v, tilt=tilt)


# --------------------------------------------------------------------------- #
#  Tooth placement
# --------------------------------------------------------------------------- #
def fs_tooth_positions(phi: float, psi_f: float, par: MeshParams,
                       exact: bool = False) -> dict:
    """Deformed position of every FS tooth at WG angle phi, FS rotation psi_f.

    Returns per-tooth arrays (length z_f):
      k      tooth index
      beta   undeformed material angle in fixed frame  = 2pi k/z_f + psi_f
      theta  angle from current major axis             = beta - phi   (wrapped)
      r      deformed radius                            = R + w(theta)
      ang    deformed angular position (fixed frame)    = beta + v/R
      tilt   tooth tilt chi(theta)
    """
    k = np.arange(par.z_f)
    beta = 2 * np.pi * k / par.z_f + psi_f
    theta = _wrap(beta - phi)
    pm = point_motion(theta, par, exact=exact)
    r = par.R + pm["w"]
    ang = beta + pm["v"] / par.R
    return dict(k=k, beta=beta, theta=theta, r=r, ang=_wrap(ang),
                w=pm["w"], v=pm["v"], tilt=pm["tilt"])


def cs_space_centers(par: MeshParams) -> np.ndarray:
    """Angular centres of the CS tooth SPACES in the fixed frame [rad].
    CS teeth sit at 2pi j / z_c; the spaces (gaps) sit half a pitch later."""
    j = np.arange(par.z_c)
    return _wrap(2 * np.pi * (j + 0.5) / par.z_c)


# --------------------------------------------------------------------------- #
#  Mesh state / engagement
# --------------------------------------------------------------------------- #
def mesh_state(phi: float, psi_f: float, par: MeshParams,
               exact: bool = False) -> dict:
    """Full per-FS-tooth mesh state at (phi, psi_f).

    For each FS tooth: nearest CS space, registration error e_k [rad], tangential
    offset t_k [mm], loaded-flank penetration delta_k [mm] resolved along the line
    of action, radial-engaged flag (w>0, the +/-45 deg wrap) and contact flag.
    """
    fs = fs_tooth_positions(phi, psi_f, par, exact=exact)
    spaces = cs_space_centers(par)

    # nearest CS space centre for every FS tooth (vectorised wrap-around search)
    d = _wrap(fs["ang"][:, None] - spaces[None, :])          # (z_f, z_c)
    jmin = np.argmin(np.abs(d), axis=1)
    e = d[np.arange(par.z_f), jmin]                          # signed reg. error [rad]

    t_off = e * fs["r"]                                      # tangential offset  [mm]
    pen = np.maximum(0.0, np.abs(t_off) - par.b_h)           # flank penetration  [mm]
    delta = pen * np.cos(par.alpha)                          # along line of action

    radial_engaged = fs["w"] > 0.0                           # |theta| < 45 deg
    contact = radial_engaged & (pen > 0.0)
    flank = np.where(t_off >= 0.0, +1, -1)                   # +lead / -trail

    return dict(**fs, space_j=jmin, e=e, t_off=t_off,
                delta=delta, radial_engaged=radial_engaged,
                contact=contact, flank=flank, spaces=spaces)


def engaged_set(phi: float, psi_f: float, par: MeshParams,
                exact: bool = False) -> dict:
    """Summary of the engaged set at (phi, psi_f)."""
    st = mesh_state(phi, psi_f, par, exact=exact)
    idx = np.where(st["contact"])[0]
    n_radial = int(st["radial_engaged"].sum())
    # split the contacting teeth into the two lobes (theta near 0 and near pi)
    lobes = _lobe_split(st["theta"][idx]) if idx.size else ([], [])
    return dict(indices=idx, n_contact=int(idx.size), n_radial=n_radial,
                lobe0=lobes[0], lobe1=lobes[1],
                delta=st["delta"][idx], theta=st["theta"][idx], state=st)


def sweep_engagement(par: MeshParams, n_phi: int = 181, psi_f=None,
                     exact: bool = False) -> dict:
    """Engaged-tooth count vs WG angle over one full WG revolution.

    psi_f: None -> lock FS to the kinematic ratio psi_f = kinematic_ratio * phi
    (rigid tooth-counting reference); or pass a scalar/array to override.
    """
    phis = np.linspace(0.0, 2 * np.pi, n_phi)
    n_contact = np.zeros(n_phi, dtype=int)
    n_radial = np.zeros(n_phi, dtype=int)
    for i, phi in enumerate(phis):
        pf = par.kinematic_psi_f(phi) if psi_f is None else (
            psi_f if np.isscalar(psi_f) else psi_f[i])
        es = engaged_set(phi, pf, par, exact=exact)
        n_contact[i] = es["n_contact"]
        n_radial[i] = es["n_radial"]
    return dict(phi=phis, n_contact=n_contact, n_radial=n_radial)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _lobe_split(theta):
    """Split contacting-tooth thetas into the two lobes (near 0, near +/-pi)."""
    near0 = np.where(np.abs(theta) < np.pi / 2)[0]
    near1 = np.where(np.abs(theta) >= np.pi / 2)[0]
    return list(near0), list(near1)


def summary(par: MeshParams, exact: bool = False) -> str:
    L = []
    L.append("=" * 70)
    L.append("STRAIN-WAVE MESH KINEMATICS (common.mesh)")
    L.append("=" * 70)
    L.append(f"z_f={par.z_f}  z_c={par.z_c}  R={par.R} mm  m={par.m:.4g} mm  "
             f"R_c={par.R_c:.4g} mm  w0={par.w0} mm  alpha={par.alpha_deg} deg")
    L.append(f"reduction ratio |z_f/(z_c-z_f)| = {abs(par.z_f/(par.z_c-par.z_f)):.0f}:1"
             f"   (d psi_f/d phi = {par.kinematic_ratio:+.4g})")
    L.append("")
    pf0 = par.kinematic_psi_f(0.0)
    st = mesh_state(0.0, pf0, par, exact=exact)
    es = engaged_set(0.0, pf0, par, exact=exact)
    L.append(f"--- snapshot at phi = 0, psi_f = phase0 = {np.rad2deg(pf0):.3f} deg ---")
    L.append(f"  teeth in radial wrap (|theta|<45 deg) = {es['n_radial']}"
             f"  (~ z_f/2 over both lobes)")
    # registration of the tooth nearest each lobe (the node quality)
    k0 = int(np.argmin(np.abs(st["theta"])))
    L.append(f"  best-registered tooth: e = {st['e'][k0]*1e6:.1f} urad, "
             f"t_off = {st['t_off'][k0]*1e3:.2f} um")
    # registration error at the wrap edge (theta ~ 45 deg)
    edge = int(np.argmin(np.abs(np.abs(st["theta"]) - np.pi / 4)))
    L.append(f"  wrap edge (|theta|~45 deg): e = {st['e'][edge]*1e6:.1f} urad, "
             f"t_off = {st['t_off'][edge]*1e3:.2f} um  "
             f"(half-pitch limit {np.pi/par.z_c*par.R_c*1e3:.1f} um)")
    L.append(f"  teeth in flank contact (backlash={par.backlash} mm) = "
             f"{es['n_contact']}  (lobe0 {len(es['lobe0'])}, lobe1 {len(es['lobe1'])})")
    sw = sweep_engagement(par, n_phi=181, exact=exact)
    L.append(f"  radial-wrap count over WG rev: min {sw['n_radial'].min()}  "
             f"max {sw['n_radial'].max()}  mean {sw['n_radial'].mean():.1f}")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    par = MeshParams()
    print(summary(par))
