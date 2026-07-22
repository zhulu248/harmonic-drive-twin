"""
Conjugate tooth-profile synthesis and relative sliding velocity (Topic 5).

Builds ON TOP of ``common.mesh`` (Topic-4-owned kinematics -- imported read-only;
this module never edits mesh.py).  It answers Topic 5:

  * synthesise the circular-spline (CS) tooth flank as the CONJUGATE ENVELOPE of a
    given flexspline (FS) tooth swept through the mesh by the elliptical (u, v,
    tilt) carrier motion, and
  * compute the RELATIVE SLIDING VELOCITY v_slide(theta) between the meshing FS
    and CS flanks, magnitude and where it peaks, feeding wear / friction.

Kinematics (see common.mesh for the carrier point motion)
---------------------------------------------------------
One FS tooth's whole mesh cycle is parameterised by the wave phase
    theta = beta - phi        (0 = major axis = deepest engagement)
The reference tooth (beta = psi_f) obeys  theta = (rho - 1) phi  with
    rho = d psi_f/d phi = -(z_c - z_f)/z_f     (mesh.kinematic_ratio, = -1/100)
so   phi = theta/(rho - 1).

Pose of the FS tooth ROOT (origin O) in the fixed CS frame, and the tooth
orientation Theta (centreline direction):
    r(theta)   = R + w(theta)
    A(theta)   = rho*phi + v(theta)/R           (angular position of O)
    O(theta)   = r (cosA, sinA)
    Theta(theta) = rho*phi + chi(theta)         (chi = tooth tilt)
A point Q rigidly attached to the tooth at body offset d has fixed-frame velocity
    v_Q = [ dO/dphi + (dTheta/dphi) z_hat x d ] * Omega_WG.
Because the CS is fixed, the sliding velocity of the FS flank over the CS flank is
just v_Q resolved along the common tangent; at a true contact point the conjugacy
condition n . v_Q = 0 holds, so v_slide = |v_Q| there.

Headline: dO/dphi is O(w0) per radian, NOT O(R).  Strain-wave sliding therefore
scales with the ovalization w0 (~0.3 mm), not the pitch radius R (20 mm): tooth
sliding is ~R/w0 ~ 60x smaller than a same-size involute gear -- the kinematic
reason harmonic drives wear slowly despite 100+ teeth in mesh.

Units: mm, rad, s.  theta = 0 is the major axis / deepest engagement.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from harmonic_drive_twin.flex_common.mesh import MeshParams, point_motion   # noqa: E402
from harmonic_drive_twin.flex_common.teeth import ToothGeom                  # noqa: E402


# --------------------------------------------------------------------------- #
#  FS tooth outline (parametric, from ToothGeom ratios)
# --------------------------------------------------------------------------- #
@dataclass
class FSTooth:
    """Flexspline tooth outline in the tooth-local frame.

    Local frame: origin on the rim neutral line at the tooth centre; x = tangential
    (+theta sense), y = radial OUTWARD.  The tooth is deliberately NARROW with a
    ROUNDED TIP (matching the physical sample): straight flanks inclined at the
    pressure angle, closed by a tip fillet of radius rho_tip.

    Shape is set by ratios to the module m so SIZE and SHAPE vary independently
    (mirrors common.teeth.ToothGeom).
    """
    m: float = 0.20            # module 2R/z                                    [mm]
    alpha_deg: float = 20.0    # flank pressure angle                          [deg]
    sF_ratio: float = 1.20     # root chord thickness s_F = sF_ratio*m         [-]
    ha_ratio: float = 1.00     # addendum (tip height above pitch) = ha_ratio*m [-]
    hd_ratio: float = 0.60     # how far below pitch the working flank starts   [-]
    rho_tip_ratio: float = 0.30  # tip fillet radius = rho_tip_ratio*m          [-]

    @property
    def alpha(self) -> float:
        return np.deg2rad(self.alpha_deg)

    @property
    def s_F(self) -> float:
        return self.sF_ratio * self.m

    @property
    def h_a(self) -> float:
        return self.ha_ratio * self.m

    @property
    def rho_tip(self) -> float:
        return self.rho_tip_ratio * self.m

    def outline(self, n: int = 400) -> dict:
        """Right-half working outline as (x, y) with OUTWARD unit normals (nx, ny).

        Returns the loaded (right) flank from the base up over the tip fillet to the
        apex.  x>0 side; mirror for the other flank.  Points ordered base -> tip.
        """
        a = self.alpha
        xb = 0.5 * self.s_F                              # half root thickness
        # straight flank: x = xb - (y+hd)*tan(alpha) measured from working start
        y0 = -self.hd_ratio * self.m                     # start a bit below pitch
        # tip fillet centre on the centreline, radius rho_tip, apex at y = h_a
        rt = self.rho_tip
        yc = self.h_a - rt                               # fillet centre height
        # flank line: point (x = xb - (y - y0) tan a, y).  It is tangent to the tip
        # circle; find the tangency height y_t by intersecting flank with circle.
        ta = np.tan(a)
        # flank direction unit (going up-inward): (-sin a, cos a); outward normal
        # (cos a, sin a).
        # solve for tangency: distance from centre (0, yc) to flank line = rt.
        # line: x*cos a + (y)*sin a = C  with C = xb*cos a + y0? re-derive:
        # x = xb - (y - y0) ta  ->  x + (y - y0) ta = xb  -> x cos a + (y-y0) sin a = xb cos a
        # signed distance of (0, yc): |0 + (yc - y0) sin a - xb cos a| ... set = rt
        # Instead sample flank, then the arc, and drop points inside the circle.
        yv = np.linspace(y0, self.h_a, n)
        xv = xb - (yv - y0) * ta
        # where the flank would cross the centreline / enter tip circle, clip:
        d_to_c = np.hypot(xv - 0.0, yv - yc)
        on_flank = (yv <= yc) | (d_to_c >= rt)
        xf, yf = xv[on_flank], yv[on_flank]
        # tip fillet arc from tangency to apex
        phi_arc = np.linspace(0.0, np.pi / 2, max(20, n // 8))
        # tangency angle on circle where flank meets it:
        # flank outward normal (cos a, sin a) -> tangency point = centre + rt*normal
        xt = 0.0 + rt * np.cos(a)
        yt = yc + rt * np.sin(a)
        ang0 = np.arctan2(yt - yc, xt - 0.0)             # ~ (pi/2 - a)
        arc_ang = np.linspace(ang0, np.pi / 2, max(20, n // 8))
        xa = rt * np.cos(arc_ang)
        ya = yc + rt * np.sin(arc_ang)
        # assemble base->tip
        x = np.concatenate([xf, xa])
        y = np.concatenate([yf, ya])
        # outward normals: flank -> (cos a, sin a); arc -> radial from centre
        nfx = np.full(xf.shape, np.cos(a))
        nfy = np.full(yf.shape, np.sin(a))
        nax = np.cos(arc_ang)
        nay = np.sin(arc_ang)
        nx = np.concatenate([nfx, nax])
        ny = np.concatenate([nfy, nay])
        nn = np.hypot(nx, ny)
        return dict(x=x, y=y, nx=nx / nn, ny=ny / nn,
                    y0=y0, yc=yc, rt=rt, xb=xb)


# --------------------------------------------------------------------------- #
#  FS tooth outline from a POINTWISE curve (measured / CAD-imported / designed)
# --------------------------------------------------------------------------- #
class PointwiseTooth:
    """Flexspline tooth working flank given POINT BY POINT (not parametrically).

    For profiles that exist only as a list of (x, y) points -- e.g. the current
    German production design, stored in Creo as an imported point spline
    (PNT_TOOTHGAP_IMPORT), or a profile generated by topic5's tooth_designer.
    Fits a parametric cubic spline through the points and exposes the same
    ``outline()`` interface as FSTooth, so ``conjugate_cs_profile`` /
    ``sliding_summary`` work on it unchanged.

    Frame convention (same as FSTooth): tooth-local, origin on the rim neutral
    line at the tooth centre, x = tangential (working flank on x > 0), y =
    radial OUTWARD.  Points must be ordered ALONG the curve (either direction);
    they are re-ordered base -> tip internally.  The curve should be the working
    (right) half: root/base end up over the flank to the tip apex.

    Parameters
    ----------
    x, y   : point coordinates in the tooth-local frame [mm]
    smooth : spline smoothing factor s of scipy splprep (0 = interpolate
             exactly; ~len(x)*sigma^2 to smooth measurement noise sigma)
    """

    def __init__(self, x, y, smooth: float = 0.0):
        from scipy.interpolate import splprep
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 4:
            raise ValueError("PointwiseTooth needs at least 4 points")
        if y[0] > y[-1]:                      # given tip -> base: flip
            x, y = x[::-1], y[::-1]
        self.px, self.py = x, y
        # chord-length parametrisation (splprep default) with optional smoothing
        self._tck, self._u = splprep([x, y], s=smooth, k=3)

    def outline(self, n: int = 400) -> dict:
        """Resampled outline base -> tip with outward unit normals (nx, ny)."""
        from scipy.interpolate import splev
        u = np.linspace(0.0, 1.0, n)
        x, y = splev(u, self._tck)
        tx, ty = splev(u, self._tck, der=1)
        tn = np.hypot(tx, ty)
        tx, ty = tx / tn, ty / tn
        # tangent points base->tip; outward normal of the right flank is the
        # tangent rotated -90 deg (matches FSTooth: t=(-sin a, cos a) ->
        # n=(cos a, sin a))
        nx, ny = ty, -tx
        return dict(x=np.asarray(x), y=np.asarray(y), nx=nx, ny=ny)


# --------------------------------------------------------------------------- #
#  Tooth pose & velocity in the fixed CS frame
# --------------------------------------------------------------------------- #
def _phi_of_theta(theta, par: MeshParams):
    rho = par.kinematic_ratio
    return theta / (rho - 1.0)


def tooth_pose(theta, par: MeshParams) -> dict:
    """Fixed-frame pose of the reference FS tooth at wave phase theta.

    Returns O (origin, shape (...,2)), A (angular position of O), Theta (tooth
    centreline orientation), and the per-unit-Omega rates dO/dphi, dTheta/dphi.
    """
    theta = np.asarray(theta, dtype=float)
    rho = par.kinematic_ratio
    R = par.R
    phi = _phi_of_theta(theta, par)
    pm = point_motion(theta, par)                       # w, v, tilt, wp
    w, v, chi, wp = pm["w"], pm["v"], pm["tilt"], pm["wp"]
    # derivatives wrt theta
    vp = -pm["w"]                                        # v' = -w (inextensible)
    chip = -(3.0 * par.w0 / R) * np.cos(2 * theta)      # d chi/d theta
    r = R + w
    A = rho * phi + v / R
    Theta = rho * phi + chi
    O = np.stack([r * np.cos(A), r * np.sin(A)], axis=-1)
    # rates: dtheta/dphi = rho - 1
    dth = rho - 1.0
    dr = wp * dth
    dA = rho + (vp / R) * dth
    dO = np.stack([dr * np.cos(A) - r * dA * np.sin(A),
                   dr * np.sin(A) + r * dA * np.cos(A)], axis=-1)
    dTheta = rho + chip * dth
    return dict(theta=theta, phi=phi, O=O, A=A, Theta=Theta,
                dO=dO, dTheta=dTheta, r=r, w=w, v=v, chi=chi)


def _to_fixed(pt_local, O, Theta):
    """Map tooth-local (x,y) to fixed frame given origin O and orientation Theta.
    Local +y is the tooth centreline (radial-out); it maps to direction Theta."""
    c, s = np.cos(Theta), np.sin(Theta)
    # local x -> tangential (Theta - pi/2 direction), local y -> Theta direction
    ex = np.array([np.sin(Theta), -np.cos(Theta)])      # local +x axis in fixed
    ey = np.array([np.cos(Theta), np.sin(Theta)])       # local +y axis in fixed
    x, y = pt_local
    return O + x * ex + y * ey


# --------------------------------------------------------------------------- #
#  CLOSED-FORM results (derived analytically, verified in derive_symbolic.py)
# --------------------------------------------------------------------------- #
def sliding_speed_base(theta, par: MeshParams) -> np.ndarray:
    """Closed-form sliding speed of the tooth ROOT (neutral-line carrier) per unit
    WG rate Omega [mm/rad].  Exact to first order in the shell displacement:

        |v|/Omega = sqrt[ (2 kappa w0 sin2θ)^2 + (R rho + kappa w0 cos2θ)^2 ]

    with kappa = z_c/z_f,  rho = -(z_c - z_f)/z_f = 1 - kappa,  R rho = -(z_c-z_f)m/2.
    Verified vs the full numeric envelope (dominant term; the contact point on the
    tip adds a small +chi'-spin correction, ~1-2%)."""
    theta = np.asarray(theta, dtype=float)
    kap = par.z_c / par.z_f
    rho = par.kinematic_ratio
    return np.sqrt((2 * kap * par.w0 * np.sin(2 * theta)) ** 2
                   + (par.R * rho + kap * par.w0 * np.cos(2 * theta)) ** 2)


def sliding_floor(par: MeshParams) -> dict:
    """Deep-engagement (theta=0) sliding floor and the w0 that nulls it.

        floor/Omega = |R rho + kappa w0| = |kappa w0 - (z_c - z_f) m / 2|
        floor -> 0   when   w0* = (z_c - z_f) m / (2 kappa) ≈ (z_c - z_f) m / 2

    So the mid-mesh sliding is minimised by matching the ovalization w0 to the
    per-tooth advance (z_c - z_f) m / 2 (a genuine design lever)."""
    kap = par.z_c / par.z_f
    rho = par.kinematic_ratio
    a_adv = (par.z_c - par.z_f) * par.m / 2.0
    return dict(floor=abs(par.R * rho + kap * par.w0),
                a_adv=a_adv, w0_opt=a_adv / kap, w0=par.w0)


def conjugate_cs_closed_form(fs: FSTooth, par: MeshParams,
                             theta_max_deg: float = 12.0,
                             n_theta: int = 121) -> dict:
    """Analytic CS conjugate flank for a circular-arc FS tip.

    Model assumption: the working surface of the FS tooth is its ROUNDED TIP, a
    circular arc of radius rho_tip centred at local (0, h_c), h_c = h_a - rho_tip.
    Then the CS conjugate flank is the ENVELOPE of that moving circle, i.e. the
    tip-centre trajectory C(theta) = O(theta) + h_c e_Y(theta) OFFSET by rho_tip
    along the trajectory normal:

        E(theta) = C(theta) + rho_tip * n_hat,   n_hat = left-normal of dC/dtheta.

    Matches the general n.v=0 envelope to <0.5% of module (see derive_symbolic.py)."""
    th = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, n_theta))
    h_c = fs.h_a - fs.rho_tip
    rt = fs.rho_tip
    d = 1e-6
    p0 = tooth_pose(th, par)
    p1 = tooth_pose(th + d, par)
    def centre(p):
        Th = p["Theta"]
        eY = np.stack([np.cos(Th), np.sin(Th)], axis=-1)
        return p["O"] + h_c * eY
    C0 = centre(p0); C1 = centre(p1)
    tang = C1 - C0
    tang /= np.hypot(tang[:, 0], tang[:, 1])[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
    E = C0 + rt * nrm
    return dict(theta=th, cs_x=E[:, 0], cs_y=E[:, 1],
                centre_x=C0[:, 0], centre_y=C0[:, 1], rho_tip=rt, h_c=h_c)


# --------------------------------------------------------------------------- #
#  Conjugate envelope: CS flank = locus of contact points (n . v = 0)
# --------------------------------------------------------------------------- #
def conjugate_cs_profile(fs: FSTooth, par: MeshParams,
                         theta_max_deg: float = 30.0, n_theta: int = 241,
                         side: int = +1) -> dict:
    """Generate the CS conjugate flank as the envelope of the moving FS tooth.

    For each wave phase theta, find the point on the FS working flank where the FS
    material velocity is tangent to the flank (n . v = 0).  That instantaneous
    contact point, mapped to the fixed CS frame, is a point on the CS conjugate
    profile.  side=+1 loaded (leading) flank, -1 trailing.

    Returns arrays over theta: contact point in CS frame (cs_x, cs_y), the flank
    parameter s* (arc index), the contact point in the FS-tooth frame (fs_x,fs_y),
    sliding speed per unit Omega (vslide), and the contact normal.
    """
    ol = fs.outline()
    # right (loaded) flank points & normals in local frame; side flips x
    lx = side * ol["x"]
    ly = ol["y"]
    lnx = side * ol["nx"]
    lny = ol["ny"]

    thetas = np.deg2rad(np.linspace(-theta_max_deg, theta_max_deg, n_theta))
    cs_x = np.full(n_theta, np.nan); cs_y = np.full(n_theta, np.nan)
    fs_x = np.full(n_theta, np.nan); fs_y = np.full(n_theta, np.nan)
    vslide = np.full(n_theta, np.nan)
    sstar = np.full(n_theta, np.nan)
    hgt = np.full(n_theta, np.nan)

    for i, th in enumerate(thetas):
        pose = tooth_pose(th, par)
        O = pose["O"]; Theta = float(pose["Theta"])
        dO = pose["dO"]; dTh = float(pose["dTheta"])
        ex = np.array([np.sin(Theta), -np.cos(Theta)])
        ey = np.array([np.cos(Theta), np.sin(Theta)])
        # fixed-frame points, normals, velocities along the flank
        P = O[None, :] + lx[:, None] * ex[None, :] + ly[:, None] * ey[None, :]
        Nf = lnx[:, None] * ex[None, :] + lny[:, None] * ey[None, :]
        d = P - O[None, :]                              # offset from origin
        zc = np.stack([-d[:, 1], d[:, 0]], axis=-1)     # z_hat x d
        V = dO[None, :] + dTh * zc                      # velocity per unit Omega
        ndotv = np.sum(Nf * V, axis=1)                  # conjugacy residual
        # find sign change along the flank -> contact point.  Multiple crossings
        # can occur (a degenerate one near the base); the physical contact rides
        # the WORKING flank/tip, so take the crossing nearest the tip (max index).
        sgn = np.sign(ndotv)
        cross = np.where(np.diff(sgn) != 0)[0]
        if cross.size == 0:
            continue
        j = cross[-1]
        # linear interpolate the zero
        f = ndotv[j] / (ndotv[j] - ndotv[j + 1])
        s_star = j + f
        px = P[j] + f * (P[j + 1] - P[j])
        vv = V[j] + f * (V[j + 1] - V[j])
        cs_x[i], cs_y[i] = px
        vslide[i] = np.hypot(*vv)
        sstar[i] = s_star
        fx = lx[j] + f * (lx[j + 1] - lx[j])
        fy = ly[j] + f * (ly[j + 1] - ly[j])
        fs_x[i], fs_y[i] = fx, fy
        hgt[i] = fy
    return dict(theta=thetas, cs_x=cs_x, cs_y=cs_y, fs_x=fs_x, fs_y=fs_y,
                vslide=vslide, sstar=sstar, height=hgt, fs_outline=ol, side=side)


# --------------------------------------------------------------------------- #
#  Curve identification: fit the CS conjugate to a circular arc, vs involute
# --------------------------------------------------------------------------- #
def fit_circle(x, y) -> dict:
    """Least-squares circle fit (Kasa).  Returns centre, radius, RMS residual."""
    x = np.asarray(x); y = np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = 0.5 * c[0], 0.5 * c[1]
    rad = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    res = np.hypot(x - xc, y - yc) - rad
    return dict(xc=xc, yc=yc, R=rad, rms=float(np.sqrt(np.mean(res ** 2))),
                n=int(x.size))


def sliding_summary(par: MeshParams, fs: FSTooth | None = None,
                    omega_wg: float = 1.0) -> dict:
    """v_slide over the mesh: peak, where, and the pitch-line comparison."""
    fs = fs or FSTooth(m=par.m, alpha_deg=par.alpha_deg)
    cj = conjugate_cs_profile(fs, par)
    vs = cj["vslide"] * omega_wg
    th = cj["theta"]
    ok = np.isfinite(vs)
    ipk = np.nanargmax(np.where(ok, vs, -np.inf))
    v_involute = par.R * omega_wg                       # same-size gear pitch speed
    return dict(theta=th, vslide=vs, v_peak=float(vs[ipk]),
                theta_peak_deg=float(np.degrees(th[ipk])),
                v_pitchline_involute=v_involute,
                ratio_involute=float(v_involute / max(1e-12, vs[ipk])),
                circle=fit_circle(cj["cs_x"], cj["cs_y"]),
                conj=cj)


if __name__ == "__main__":
    from common.baseline import BASELINE as B
    par = MeshParams(z_f=B.z_flex, z_c=B.z_circ, R=B.R, w0=B.w0,
                     alpha_deg=B.alpha_deg)
    fs = FSTooth(m=par.m, alpha_deg=par.alpha_deg)
    print(f"profile.py self-test  m={par.m:.4g} mm  alpha={par.alpha_deg} deg  "
          f"w0={par.w0} mm")
    ss = sliding_summary(par, fs)
    print(f"  v_slide peak      = {ss['v_peak']:.4g} mm/rad  "
          f"at theta = {ss['theta_peak_deg']:+.1f} deg")
    print(f"  same-size involute pitch speed R*Omega = "
          f"{ss['v_pitchline_involute']:.4g} mm/rad")
    print(f"  sliding reduction vs involute          = "
          f"{ss['ratio_involute']:.1f}x smaller")
    c = ss["circle"]
    print(f"  CS conjugate circle fit: R={c['R']:.4g} mm  rms={c['rms']*1e3:.3g} um "
          f"({c['n']} pts)  -> {'near-circular arc' if c['rms']<0.02 else 'non-circular'}")
