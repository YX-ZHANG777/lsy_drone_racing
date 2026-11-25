"""Observation features and action scaling for the racing RL policy.

Written so that the same code runs with numpy (deployment) and jax.numpy (training):
pass the array module as ``xp``. All inputs are batched: leading dim = n_envs.

Feature design follows Song et al. 2021/2023 (gate-corner observations, previous action):
  own state        pos(3) vel(3) rotmat(9) ang_vel(3) prev. vel(3)        = 21
  next gate        corners rel. to drone (4x3) + signed normal (3)        = 15
                   drone pos + vel in the gate frame (x along pass dir)   =  6
  gate after next  corners rel. to drone (4x3) + signed normal (3)        = 15
  obstacles        nearest 2, rel. xy (2x2)                               =  4
  last action      (3)                                                    =  3
  previous gate    drone pos + vel in the frame of the gate just passed   =  6
                   (needed to clear the frame before turning; [2,0,0],0 before the first gate)
  obstacle CPA     nearest 2 poles: xy distance, closest-approach distance
                   and time to closest approach at the current velocity   =  6
  speed command    desired speed limit v_d / 3 (speed level, set by the user)  =  1
                                                                    total = 77
"""

import math

GATE_HALF = 0.2  # gate opening is 0.4 m wide (square)
N_OBS = 77
N_ACT = 3  # roll, pitch, collective thrust (yaw command fixed to 0)

# action limits
MAX_TILT = 0.75  # rad, roll/pitch command limit
ANGV_WORLD = [False]  # features_v2: rotate the body-frame angular velocity to the world frame first (fix, 11-19)


def quat_to_rotmat(q, xp):
    """xyzw quaternion (..., 4) -> rotation matrix (..., 3, 3)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return xp.stack(
        [xp.stack([r00, r01, r02], -1), xp.stack([r10, r11, r12], -1), xp.stack([r20, r21, r22], -1)],
        -2,
    )


_CORNERS = ((0.0, 1.0, 1.0), (0.0, -1.0, 1.0), (0.0, -1.0, -1.0), (0.0, 1.0, -1.0))


def gate_targets(obs, xp):
    """Indices (into gates) and directions of the next two gates in the sequence."""
    seq = obs["gate_sequence"]  # (N, S)
    sdir = obs["gate_sequence_direction"]  # (N, S)
    n_seq = seq.shape[-1]
    k1 = xp.clip(obs["n_gates_passed"], 0, n_seq - 1)
    k2 = xp.clip(k1 + 1, 0, n_seq - 1)
    ar = xp.arange(seq.shape[0])
    return seq[ar, k1], sdir[ar, k1], seq[ar, k2], sdir[ar, k2]


def _gate_feats(obs, gid, gdir, xp):
    ar = xp.arange(gid.shape[0])
    c = obs["gates_pos"][ar, gid]  # (N,3)
    Rg = quat_to_rotmat(obs["gates_quat"][ar, gid], xp)  # (N,3,3)
    corners = xp.asarray(_CORNERS, dtype=c.dtype) * GATE_HALF  # (4,3)
    cw = c[:, None, :] + xp.einsum("nij,kj->nki", Rg, corners)  # (N,4,3)
    rel = cw - obs["pos"][:, None, :]
    normal = Rg[:, :, 0] * gdir[:, None].astype(c.dtype)
    return rel.reshape(rel.shape[0], 12), normal


def gate_local(obs, xp, prev=False):
    """Drone position and velocity in the frame of the next gate (x = passing direction).
    With ``prev=True``: in the frame of the gate passed last."""
    g1, d1, _, _ = gate_targets(obs, xp)
    if prev:
        k0 = xp.clip(obs["n_gates_passed"] - 1, 0, obs["gate_sequence"].shape[-1] - 1)
        ar_ = xp.arange(g1.shape[0])
        g1 = obs["gate_sequence"][ar_, k0]
        d1 = obs["gate_sequence_direction"][ar_, k0]
    ar = xp.arange(g1.shape[0])
    c = obs["gates_pos"][ar, g1]
    Rg = quat_to_rotmat(obs["gates_quat"][ar, g1], xp)
    sgn = xp.stack([d1, d1, xp.ones_like(d1)], -1).astype(c.dtype)  # flip x,y for reverse passes
    lp = xp.einsum("nji,nj->ni", Rg, obs["pos"] - c) * sgn
    lv = xp.einsum("nji,nj->ni", Rg, obs["vel"]) * sgn
    return lp, lv


def target_center(obs, xp):
    g1, d1, _, _ = gate_targets(obs, xp)
    ar = xp.arange(g1.shape[0])
    return obs["gates_pos"][ar, g1]


def features(obs, last_action, prev_vel, xp, vlim=None):
    """Build the policy input (N, N_OBS) from a batched observation dict."""
    pos, vel, angv = obs["pos"], obs["vel"], obs["ang_vel"]
    rot = quat_to_rotmat(obs["quat"], xp).reshape(pos.shape[0], 9)
    g1, d1, g2, d2 = gate_targets(obs, xp)
    c1, n1 = _gate_feats(obs, g1, d1, xp)
    c2, n2 = _gate_feats(obs, g2, d2, xp)
    dxy = obs["obstacles_pos"][:, :, :2] - pos[:, None, :2]  # (N,M,2)
    dist = xp.sqrt((dxy**2).sum(-1))
    order = xp.argsort(dist, axis=-1)[:, :2]
    near2 = xp.take_along_axis(dxy, order[:, :, None], axis=1)  # (N,2,2) obstacle - drone
    near = near2.reshape(pos.shape[0], 4)
    vxy = vel[:, None, :2]
    v2 = (vxy**2).sum(-1) + 1e-3
    t_cpa = xp.clip((near2 * vxy).sum(-1) / v2, 0.0, 2.0)  # time until closest approach (s)
    miss = xp.sqrt(((near2 - vxy * t_cpa[..., None]) ** 2).sum(-1))  # closest-approach distance
    dist2 = xp.sqrt((near2**2).sum(-1))
    cpa = xp.concatenate([xp.minimum(dist2, 2.0), xp.minimum(miss, 2.0), t_cpa], -1)
    lp, lv = gate_local(obs, xp)
    pp, pv = gate_local(obs, xp, prev=True)
    first = (obs["n_gates_passed"] == 0)[:, None]
    pp = xp.where(first, xp.asarray([2.0, 0.0, 0.0], dtype=pos.dtype), pp)
    pv = xp.where(first, xp.zeros_like(pv), pv)
    return xp.concatenate(
        [pos, vel, rot, angv, prev_vel, c1, n1, lp, lv, c2, n2, near, last_action, pp, pv, cpa,
         (xp.full((pos.shape[0], 1), 4.0, dtype=pos.dtype) if vlim is None
          else xp.reshape(vlim, (-1, 1)).astype(pos.dtype)) / 3.0],
        -1,
    )


def scale_action(a, thrust_min, thrust_max, xp):
    """Normalized action in [-1,1]^3 -> env attitude command [roll, pitch, yaw, thrust]."""
    a = xp.clip(a, -1.0, 1.0)
    roll = a[..., 0] * MAX_TILT
    pitch = a[..., 1] * MAX_TILT
    yaw = xp.zeros_like(roll)
    thrust = thrust_min + (a[..., 2] + 1.0) * 0.5 * (thrust_max - thrust_min)
    return xp.stack([roll, pitch, yaw, thrust], -1)


# ---------------------------------------------------------------------------------------------
# Version 2: canonical "gate frame" (rotation invariant). All horizontal vectors are expressed in
# a frame whose x axis is the horizontal passing direction of the next gate; the policy outputs
# the desired tilt in that frame, which is rotated back to world roll/pitch (yaw command = 0).
N_OBS2 = 77
N_ACT2 = 3


def frame_yaw(obs, xp):
    """Yaw of the canonical frame = horizontal passing direction of the next gate."""
    g1, d1, _, _ = gate_targets(obs, xp)
    ar = xp.arange(g1.shape[0])
    Rg = quat_to_rotmat(obs["gates_quat"][ar, g1], xp)
    nx = Rg[:, 0, 0] * d1
    ny = Rg[:, 1, 0] * d1
    return xp.arctan2(ny, nx)


def _rot(v, c, s):
    """Rotate world vectors (..., 3 or 2) into the frame with yaw (cos c, sin s): R_z(-yaw) v."""
    x, y = v[..., 0], v[..., 1]
    xr, yr = c * x + s * y, -s * x + c * y
    if v.shape[-1] == 2:
        return xp_stack([xr, yr], v)
    return xp_stack([xr, yr, v[..., 2]], v)


def xp_stack(parts, like):
    import numpy as _np
    mod = _np if isinstance(like, _np.ndarray) else __import__("jax.numpy", fromlist=["x"])
    return mod.stack(parts, -1)


def features_v2(obs, last_action, prev_vel, xp, vlim=None):
    n = obs["pos"].shape[0]
    yaw = frame_yaw(obs, xp)
    c, s = xp.cos(yaw), xp.sin(yaw)
    c1d, s1d = c, s  # for (n, 3) vectors
    pos, vel = obs["pos"], obs["vel"]
    R = quat_to_rotmat(obs["quat"], xp)  # (n,3,3) body->world
    # rotate the body axes into the canonical frame
    Rh = xp.stack([_rot(R[:, :, j], c1d, s1d) for j in range(3)], -1).reshape(n, 9)
    vel_h = _rot(vel, c1d, s1d)
    pvel_h = _rot(prev_vel, c1d, s1d)
    # obs["ang_vel"] is BODY-frame (crazyflow and the real drone-estimators). Old policies (ANGV_WORLD off) rotated
    # it like a world vector; with ANGV_WORLD on it is first rotated to the world frame (consistent with vel_h).
    angv = xp.einsum("nij,nj->ni", R, obs["ang_vel"]) if ANGV_WORLD[0] else obs["ang_vel"]
    angv_h = _rot(angv, c1d, s1d)
    g1, d1, g2, d2 = gate_targets(obs, xp)
    cw1, nw1 = _gate_feats(obs, g1, d1, xp)
    cw2, nw2 = _gate_feats(obs, g2, d2, xp)
    c1h = _rot(cw1.reshape(n, 4, 3), c[:, None], s[:, None]).reshape(n, 12)
    c2h = _rot(cw2.reshape(n, 4, 3), c[:, None], s[:, None]).reshape(n, 12)
    n2h = _rot(nw2, c1d, s1d)
    n1z = nw1[:, 2:3]  # horizontal part of n1 is (1, 0) by construction
    lp, lv = gate_local(obs, xp)
    pp, pv = gate_local(obs, xp, prev=True)
    first = (obs["n_gates_passed"] == 0)[:, None]
    pp = xp.where(first, xp.asarray([2.0, 0.0, 0.0], dtype=pos.dtype), pp)
    pv = xp.where(first, xp.zeros_like(pv), pv)
    dxy = obs["obstacles_pos"][:, :, :2] - pos[:, None, :2]
    dist = xp.sqrt((dxy**2).sum(-1))
    order = xp.argsort(dist, axis=-1)[:, :2]
    near2 = xp.take_along_axis(dxy, order[:, :, None], axis=1)
    near2h = _rot(near2, c[:, None], s[:, None])  # (n,2,2)
    vxy = vel_h[:, None, :2]
    v2 = (vxy**2).sum(-1) + 1e-3
    t_cpa = xp.clip((near2h * vxy).sum(-1) / v2, 0.0, 2.0)
    miss_vec = near2h - vxy * t_cpa[..., None]  # signed closest-approach offset (canonical frame)
    vl = (xp.full((n, 1), 4.0, dtype=pos.dtype) if vlim is None
          else xp.reshape(vlim, (-1, 1)).astype(pos.dtype))
    return xp.concatenate(
        [
            pos,  # 3 world position (arena bounds)
            xp.stack([c, s], -1),  # 2 frame heading (lets the policy know the world orientation)
            vel_h, pvel_h, Rh, angv_h,  # 3+3+9+3
            c1h, n1z, lp, lv,  # 12+1+3+3
            c2h, n2h,  # 12+3
            near2h.reshape(n, 4), miss_vec.reshape(n, 4), t_cpa,  # 4+4+2
            last_action,  # 3
            pp, pv,  # 3+3
            vl / 3.0,  # 1
        ],
        -1,
    )


def scale_action_v2(a, yaw, thrust_min, thrust_max, xp):
    """Policy action [tilt_x, tilt_y, thrust] in the canonical frame -> world attitude command."""
    a = xp.clip(a, -1.0, 1.0)
    tx_h, ty_h = xp.tan(a[..., 0] * MAX_TILT), xp.tan(a[..., 1] * MAX_TILT)
    c, s = xp.cos(yaw), xp.sin(yaw)
    tx, ty = c * tx_h - s * ty_h, s * tx_h + c * ty_h  # world horizontal tilt vector
    nrm = xp.sqrt(1.0 + tx**2 + ty**2)
    dx, dy, dz = tx / nrm, ty / nrm, 1.0 / nrm  # desired thrust direction (world)
    roll = -xp.arcsin(xp.clip(dy, -1.0, 1.0))
    pitch = xp.arctan2(dx, dz)
    thrust = thrust_min + (a[..., 2] + 1.0) * 0.5 * (thrust_max - thrust_min)
    return xp.stack([roll, pitch, xp.zeros_like(roll), thrust], -1)


# ---------------------------------------------------------------------------------------------
# Safety filter: exponential control barrier function for the vertical poles (cylinders).
# Horizontal acceleration a (world, m/s^2) is projected so that for every pole
#   h = d - r_safe,  d = |p_xy - o_xy|,   h_dd + k1 h_d + k0 h >= 0
# with h_d = n.v, h_dd = (|v|^2 - h_d^2)/d + n.a  (n = (p - o)/d)  ->  n.a >= b.
# Applied sequentially to the poles (nearest last). Ref.: Ames et al. 2019 (CBF), Nguyen & Sreenath
# 2016 (exponential CBF), Cheng et al. 2019 (RL + CBF).
CBF = dict(r_safe=0.16, k1=8.0, k0=16.0, z_top=1.65, a_max=9.0,
           gate_posts=False, post_y=0.28, r_post=0.17, post_dz=0.45,
           corridor=False, corr_y=0.12, corr_z=0.16, corr_x=1.0)


def _cylinders(obs, xp, prm):
    """All vertical cylinders to avoid: poles + the two side bars of every gate.
    Returns centres (n, K, 2), radii (n, K) and active masks (n, K)."""
    p = obs["pos"]
    o = obs["obstacles_pos"]
    n, M = o.shape[0], o.shape[1]
    cen = [o[..., :2]]
    rad = [xp.full((n, M), prm["r_safe"])]
    if prm.get("r_unseen", 0.0) > 0 and "obstacles_visited" in obs:
        # pole not yet seen: its reported (nominal) position may be off by up to ~0.15 m -> inflate
        rad = [prm["r_safe"] + prm["r_unseen"] * (1.0 - obs["obstacles_visited"].astype(p.dtype))]
    act = [xp.broadcast_to((p[:, 2] < prm["z_top"])[:, None], (n, M))]
    if prm.get("gate_posts", True):
        gp, gq = obs["gates_pos"], obs["gates_quat"]
        G = gp.shape[1]
        Rg = quat_to_rotmat(gq.reshape(n * G, 4), xp).reshape(n, G, 3, 3)
        lat = Rg[:, :, :2, 1]  # gate y axis (horizontal part)
        near_z = xp.abs(p[:, None, 2] - gp[..., 2]) < prm["post_dz"]
        if prm.get("corridor", False):
            # aligned with the opening of a gate: its side bars cannot be hit -> no braking for them
            rel = p[:, None, :] - gp  # (n,G,3)
            loc = xp.einsum("ngji,ngj->ngi", Rg, rel)  # gate-local coordinates
            widen = prm.get("corr_slope", 0.0) * xp.maximum(0.0, xp.abs(loc[..., 0]) - 0.15)
            in_corr = ((xp.abs(loc[..., 1]) < prm["corr_y"] + widen) & (xp.abs(loc[..., 2]) < prm["corr_z"] + widen)
                       & (xp.abs(loc[..., 0]) < prm["corr_x"]))
            tau = prm.get("corr_tau", 0.0)
            if tau > 0:
                # velocity-aware: the position predicted tau seconds ahead must still be in the corridor
                # (otherwise a drone sliding sideways right after the gate gets the post barrier too late)
                lvel = xp.einsum("ngji,nj->ngi", Rg, obs["vel"])
                lp = loc + tau * lvel
                in_corr = in_corr & (xp.abs(lp[..., 1]) < prm["corr_y"]) & (xp.abs(lp[..., 2]) < prm["corr_z"])
            near_z = near_z & ~in_corr
        # post_seg3: each side bar (thin plate, gate-local y 0.20..0.36, x +-0.01) as 3 small circles
        # along the bar instead of one big cylinder; default: one cylinder at post_y
        offs = prm["post_offs"] if prm.get("post_seg3", False) else (prm["post_y"],)
        r_post = xp.full((n, G), prm["r_post"])
        if prm.get("box", 0.0) > 0:
            # the race env collides with a BOX (half 0.07 x 0.07 x 0.02, rotating with the drone), not the 0.086 m
            # sphere the radii were designed for: its half-width along the gate's lateral axis is
            # 0.07 (|bx.u| + |by.u|) with the drone's horizontal body axes bx, by -> 0.070 (aligned) .. 0.099 (45 deg)
            Rd = quat_to_rotmat(obs["quat"], xp)
            yd = xp.arctan2(Rd[:, 1, 0], Rd[:, 0, 0])
            un = lat / xp.sqrt((lat**2).sum(-1, keepdims=True) + 1e-9)
            cb, sb = xp.cos(yd)[:, None], xp.sin(yd)[:, None]
            ext = 0.07 * (xp.abs(cb * un[..., 0] + sb * un[..., 1]) + xp.abs(-sb * un[..., 0] + cb * un[..., 1]))
            r_post = r_post + ext - 0.086 + prm.get("box_extra", 0.0)
        for sgn in (1.0, -1.0):
            for oy in offs:
                cen.append(gp[..., :2] + sgn * oy * lat)
                rad.append(r_post)
                act.append(near_z)
    if prm.get("stand", 0.0) > 0:
        # gate stands: vertical 0.1 x 0.1 m box from the ground up to the lower frame edge
        # (gate-local z = -0.36); active while the drone (radius 0.086 m) is below that edge + margin
        gp = obs["gates_pos"]
        cen.append(gp[..., :2])
        rad.append(xp.full(gp.shape[:2], prm["stand"]))
        act.append(p[:, None, 2] < gp[..., 2] - 0.36 + prm.get("stand_dz", 0.12))
    return xp.concatenate(cen, 1), xp.concatenate(rad, 1), xp.concatenate(act, 1)


def _pred_obs(obs, prm):
    """Barrier evaluated at the position predicted ``delay`` s ahead (attitude/thrust lag).
    ``delay`` may be a scalar or a per-world array (n,) (delay-conditioned filter in training)."""
    lag = prm.get("lag_acc", 0.0)
    if lag > 0 and "acc" in obs:
        # attitude lag: the current acceleration (from attitude and thrust) persists for ~lag s
        # whatever is commanded now -> evaluate the barrier at the state lag s ahead
        acc = obs["acc"]
        obs = {**obs, "pos": obs["pos"] + 0.5 * lag**2 * acc, "vel": obs["vel"] + lag * acc}
    dl = prm.get("delay", 0.0)
    if isinstance(dl, (int, float)):
        if dl <= 0:
            return obs
        return {**obs, "pos": obs["pos"] + dl * obs["vel"]}
    return {**obs, "pos": obs["pos"] + dl[:, None] * obs["vel"]}


CBF.update(box=0.0, box_extra=0.0, stand=0.0, stand_dz=0.12, post_seg3=False, post_offs=(0.22, 0.28, 0.34), pred_T=0.0, pred_g=8.0, lag_acc=0.0,
           wall=0.0, wall_x=2.5, wall_y=1.5)


def att_acc(quat, thrust, mass, xp):
    """Acceleration produced by the current attitude and (estimated) collective thrust [N]."""
    zb = quat_to_rotmat(quat, xp)[..., :, 2]
    return zb * (thrust / mass)[..., None] - xp.asarray([0.0, 0.0, 9.81], dtype=zb.dtype)


def cbf_filter_pred(a_xy, obs, xp, prm=CBF):
    """Predictive (look-ahead) barrier for vertical cylinders.

    h = closest distance of the straight-line (constant-velocity) path over the next T seconds to
    the cylinder axis, minus its radius. A path that passes the cylinder (even closely) has h > 0 and
    is left alone; a path that would hit it within T is corrected (lateral push or braking).
    tau* = clip(-(q.v)/|v|^2, 0, T), m = q + v tau*, n = m/|m|; by the envelope theorem
    h_dot = n.(v + a tau*)  ->  constraint  tau* n.a >= -gamma h - n.v   (relative degree 1 in a).
    tau* = 0 (moving away): h_dot = n.v >= 0, nothing to do.
    """
    obs = _pred_obs(obs, prm)
    p = obs["pos"]
    v = obs["vel"]
    cen, rad, act = _cylinders(obs, xp, prm)
    T, gam = prm["pred_T"], prm["pred_g"]
    q_all = p[:, None, :2] - cen  # cylinder -> drone
    d = xp.sqrt((q_all**2).sum(-1)) + 1e-6
    order = xp.argsort(-d, axis=-1)  # farthest first, nearest last
    vxy = v[:, :2]
    vv = (vxy**2).sum(-1) + 1e-6
    for k in range(cen.shape[1]):
        j = order[:, k]
        q = xp.take_along_axis(q_all, j[:, None, None], 1)[:, 0]
        dj = xp.take_along_axis(d, j[:, None], 1)[:, 0]
        rj = xp.take_along_axis(rad, j[:, None], 1)[:, 0]
        aj = xp.take_along_axis(act, j[:, None], 1)[:, 0]
        ts = xp.clip(-(q * vxy).sum(-1) / vv, 0.0, T)
        m = q + vxy * ts[:, None]
        dm = xp.sqrt((m**2).sum(-1)) + 1e-6
        nm = m / dm[:, None]
        h = dm - rj
        lhs = ts * (nm * a_xy).sum(-1)
        b = -gam * h - (nm * vxy).sum(-1)
        active = aj & (dj < 1.5) & (ts > 1e-3)
        # minimum-norm correction along n: tau* n.(a + c n) = b
        corr = xp.where(active & (lhs < b), (b - lhs) / xp.maximum(ts, 1e-3), 0.0)
        a_xy = a_xy + corr[:, None] * nm
    if prm.get("wall", 0.0) > 0:  # arena walls (x, y limits of the official env) as predictive barriers
        for ax, lim in ((0, prm["wall_x"]), (1, prm["wall_y"])):
            for sg in (1.0, -1.0):
                vn = sg * vxy[:, ax]  # speed towards this wall
                h = lim - prm["wall"] - sg * p[:, ax] - T * xp.maximum(vn, 0.0)
                an = sg * a_xy[:, ax]
                ub = (gam * h - vn) / T  # T a_n <= gamma h - v_n
                corr = xp.where((vn > 0) & (an > ub), an - ub, 0.0)
                a_xy = a_xy - (sg * corr)[:, None] * xp.asarray([1.0 - ax, float(ax)], dtype=a_xy.dtype)
    nrm = xp.sqrt((a_xy**2).sum(-1, keepdims=True)) + 1e-9
    return a_xy * xp.minimum(1.0, prm["a_max"] / nrm)


def cbf_filter(a_xy, obs, xp, prm=CBF):
    if prm.get("pred_T", 0.0) > 0:
        return cbf_filter_pred(a_xy, obs, xp, prm)
    obs = _pred_obs(obs, prm)
    p = obs["pos"]
    v = obs["vel"]
    cen, rad, act = _cylinders(obs, xp, prm)
    dxy = p[:, None, :2] - cen
    d = xp.sqrt((dxy**2).sum(-1)) + 1e-6  # (n, K)
    order = xp.argsort(-d, axis=-1)  # farthest first, nearest last
    vxy = v[:, :2]
    for k in range(cen.shape[1]):
        j = order[:, k]
        dj = xp.take_along_axis(d, j[:, None], 1)[:, 0]
        rj = xp.take_along_axis(rad, j[:, None], 1)[:, 0]
        aj = xp.take_along_axis(act, j[:, None], 1)[:, 0]
        nj = xp.take_along_axis(dxy, j[:, None, None], 1)[:, 0] / dj[:, None]
        hd = (nj * vxy).sum(-1)
        h = dj - rj
        b = -prm["k1"] * hd - prm["k0"] * h - ((vxy**2).sum(-1) - hd**2) / dj
        active = aj & (dj < 1.5)
        na = (nj * a_xy).sum(-1)
        corr = xp.where(active & (na < b), b - na, 0.0)
        a_xy = a_xy + corr[:, None] * nj
    nrm = xp.sqrt((a_xy**2).sum(-1, keepdims=True)) + 1e-9
    return a_xy * xp.minimum(1.0, prm["a_max"] / nrm)


USE_GATE_CBF = [False]  # gate funnel filter (off: made things worse, see update log)


CBF.update(v_cap=0.0, k_v=3.0)


def speed_cap(a_xy, obs, xp, prm=CBF):
    """Horizontal speed limit as a first-order CBF: h = v_cap - |v_xy|, h_dot >= -k_v h
    <=> n.a <= k_v (v_cap - |v_xy|) with n = v_xy / |v_xy| (one linear constraint, closed-form)."""
    v = obs["vel"][..., :2]
    sp = xp.sqrt((v**2).sum(-1)) + 1e-6
    nv = v / sp[..., None]
    viol = (nv * a_xy).sum(-1) - prm["k_v"] * (prm["v_cap"] - sp)
    return a_xy - xp.maximum(viol, 0.0)[..., None] * nv


def scale_action_v2_safe(a, obs, thrust_min, thrust_max, xp, prm=CBF):
    """scale_action_v2 followed by the pole CBF filter. Returns (command, |correction| in m/s^2)."""
    a = xp.clip(a, -1.0, 1.0)
    yaw = frame_yaw(obs, xp)
    tx_h, ty_h = xp.tan(a[..., 0] * MAX_TILT), xp.tan(a[..., 1] * MAX_TILT)
    c, s = xp.cos(yaw), xp.sin(yaw)
    g = 9.81
    a_xy = g * xp.stack([c * tx_h - s * ty_h, s * tx_h + c * ty_h], -1)
    a_f = gate_filter(a_xy, obs, xp) if USE_GATE_CBF[0] else a_xy
    if prm.get("v_cap", 0.0) > 0:  # speed level for cautious flights (applied before the obstacle barriers)
        a_f = speed_cap(a_f, obs, xp, prm)
    a_f = cbf_filter_cone(a_f, obs, xp, prm) if CBF.get("cone") else cbf_filter(a_f, obs, xp, prm)
    thrust = thrust_min + (a[..., 2] + 1.0) * 0.5 * (thrust_max - thrust_min)
    corr = xp.sqrt(((a_f - a_xy) ** 2).sum(-1))
    if prm.get("bars", False):
        a_f2, thrust, dz_corr = bar_filter(a_f, thrust, obs, xp, prm, thrust_min, thrust_max)
        corr = xp.sqrt(((a_f2 - a_xy) ** 2).sum(-1) + dz_corr**2)
        a_f = a_f2
    if prm.get("ground_z", 0.0) > 0:  # take-off: limited tilt close to the floor (no tipping over)
        a_full = g * math.tan(MAX_TILT)
        lim = prm["ground_a"] + xp.clip(obs["pos"][:, 2] / prm["ground_z"], 0.0, 1.0) * (a_full - prm["ground_a"])
        nrm_a = xp.sqrt((a_f**2).sum(-1)) + 1e-9
        a_f = a_f * xp.minimum(1.0, lim / nrm_a)[:, None]
    tx, ty = a_f[:, 0] / g, a_f[:, 1] / g
    nrm = xp.sqrt(1.0 + tx**2 + ty**2)
    dx, dy, dz = tx / nrm, ty / nrm, 1.0 / nrm
    roll = -xp.arcsin(xp.clip(dy, -1.0, 1.0))
    pitch = xp.arctan2(dx, dz)
    return xp.stack([roll, pitch, xp.zeros_like(roll), thrust], -1), corr


# Top / bottom bars of every gate as horizontal cylinders (axis = gate y axis, centre at
# z_local = +-bar_z). Same exponential CBF as for the poles, but in 3D: the correction acts on the
# full acceleration (horizontal + vertical); the weighting W^-1 = diag(bar_wh, bar_wh, 1) makes it
# prefer the vertical axis (= thrust), so the filter climbs/descends instead of braking.
# Vertical acceleration a_z = T cos(tilt) / m - g  (m = nominal mass, set by env / controller).
CBF.update(bars=False, bar_z=0.28, r_bar=0.12, bar_len=0.40, bar_wh=0.1, mass=0.0,
           bar_seg3=False, bar_offs=(0.22, 0.28, 0.34), r_bar3=0.11, bar_len3=0.45)


def bar_filter(a_xy, T, obs, xp, prm, thrust_min, thrust_max):
    obs = _pred_obs(obs, prm)
    g = 9.81
    m = prm["mass"]
    p, v = obs["pos"], obs["vel"]
    n = p.shape[0]
    cos_t = 1.0 / xp.sqrt(1.0 + (a_xy**2).sum(-1) / g**2)
    az0 = T * cos_t / m - g
    acc = xp.concatenate([a_xy, az0[:, None]], -1)  # (n,3)
    gp, gq = obs["gates_pos"], obs["gates_quat"]
    G = gp.shape[1]
    Rg = quat_to_rotmat(gq.reshape(n * G, 4), xp).reshape(n, G, 3, 3)
    u = Rg[..., 1]  # bar axis (world), (n,G,3)
    zax = Rg[..., 2]
    loc = xp.einsum("ngji,ngj->ngi", Rg, p[:, None, :] - gp)
    seg3 = prm.get("bar_seg3", False)  # bars (thin plates, gate-local z 0.20..0.36) as 3 cylinders each
    near = (xp.abs(loc[..., 1]) < (prm["bar_len3"] if seg3 else prm["bar_len"])) & (xp.abs(loc[..., 0]) < 1.0)
    w = xp.asarray([prm["bar_wh"], prm["bar_wh"], 1.0], dtype=p.dtype)
    rb = prm["r_bar3"] if seg3 else prm["r_bar"]
    for sgn, oz in [(sg, o) for sg in (1.0, -1.0) for o in (prm["bar_offs"] if seg3 else (prm["bar_z"],))]:
        cen = gp + sgn * oz * zax  # (n,G,3)
        q = p[:, None, :] - cen
        q = q - (q * u).sum(-1, keepdims=True) * u
        d = xp.sqrt((q**2).sum(-1)) + 1e-6
        nv = q / d[..., None]
        vp = v[:, None, :] - (v[:, None, :] * u).sum(-1, keepdims=True) * u
        if prm.get("bar_pred", False) and prm.get("pred_T", 0.0) > 0:
            # predictive barrier (see cbf_filter_pred): closest approach of the straight-line path
            ts = xp.clip(-(q * vp).sum(-1) / ((vp**2).sum(-1) + 1e-6), 0.0, prm["pred_T"])  # (n,G)
            mm = q + vp * ts[..., None]
            dm = xp.sqrt((mm**2).sum(-1)) + 1e-6
            nm = mm / dm[..., None]
            bp = -prm["pred_g"] * (dm - rb) - (nm * vp).sum(-1)
            for j in range(G):
                nj = nm[:, j]
                lhs = ts[:, j] * (nj * acc).sum(-1)
                den = ts[:, j] * (nj**2 * w).sum(-1) + 1e-9
                ok = near[:, j] & (d[:, j] < 1.0) & (ts[:, j] > 1e-3) & (lhs < bp[:, j])
                c = xp.where(ok, (bp[:, j] - lhs) / xp.maximum(den, 1e-4), 0.0)
                acc = acc + c[:, None] * w * nj
            continue
        hd = (nv * vp).sum(-1)
        h = d - rb
        b = -prm["k1"] * hd - prm["k0"] * h - ((vp**2).sum(-1) - hd**2) / d
        for j in range(G):
            nj = nv[:, j]
            na = (nj * acc).sum(-1)
            den = (nj**2 * w).sum(-1) + 1e-9
            c = xp.where(near[:, j] & (d[:, j] < 1.0) & (na < b[:, j]), (b[:, j] - na) / den, 0.0)
            acc = acc + c[:, None] * w * nj
    a_new = acc[:, :2]
    nrm = xp.sqrt((a_new**2).sum(-1, keepdims=True)) + 1e-9
    a_new = a_new * xp.minimum(1.0, prm["a_max"] / nrm)
    cos_n = 1.0 / xp.sqrt(1.0 + (a_new**2).sum(-1) / g**2)
    T_new = xp.clip(m * (acc[:, 2] + g) / cos_n, thrust_min, thrust_max)
    return a_new, T_new, acc[:, 2] - az0


# Gate funnel barrier: near a gate the lateral offset |y| (gate frame) must stay inside a funnel
#   w(x) = y_tol + slope * max(0, -x - x0)   (x: along the passing direction, < 0 before the gate)
# h = w(x) - sgn(y) y,  h_dd = w'(x) a_x - sgn(y) a_y  ->  linear constraint on the horizontal accel.
GATE_CBF = dict(y_tol=0.10, slope=0.8, x0=0.1, x_far=1.0, x_end=0.15, k1=8.0, k0=16.0, band=0.25)


def _funnel(a_xy, p, v, c, nrm_xy, xp, prm, x_lo, x_hi, need_approach):
    nn = nrm_xy / (xp.sqrt((nrm_xy**2).sum(-1, keepdims=True)) + 1e-9)
    lat = xp.stack([-nn[:, 1], nn[:, 0]], -1)
    r = p[:, :2] - c[:, :2]
    x, y = (r * nn).sum(-1), (r * lat).sum(-1)
    vx, vy = (v[:, :2] * nn).sum(-1), (v[:, :2] * lat).sum(-1)
    sg = xp.where(y >= 0, 1.0, -1.0)
    far = -x - prm["x0"]
    w = prm["y_tol"] + prm["slope"] * xp.maximum(0.0, far)
    dw = xp.where(far > 0, -prm["slope"], 0.0)  # dw/dx
    h = w - sg * y
    hd = dw * vx - sg * vy
    gvec = dw[:, None] * nn - sg[:, None] * lat
    b = -prm["k1"] * hd - prm["k0"] * h
    active = (x > x_lo) & (x < x_hi) & (h > -prm["band"])
    if need_approach:
        active = active & (vx > 0.3)
    ga = (gvec * a_xy).sum(-1)
    g2 = (gvec**2).sum(-1) + 1e-9
    corr = xp.where(active & (ga < b), (b - ga) / g2, 0.0)
    return a_xy + corr[:, None] * gvec


def gate_filter(a_xy, obs, xp, prm=GATE_CBF):
    p, v = obs["pos"], obs["vel"]
    n = p.shape[0]
    ar = xp.arange(n)
    g1, d1, _, _ = gate_targets(obs, xp)
    c1 = obs["gates_pos"][ar, g1]
    n1 = quat_to_rotmat(obs["gates_quat"][ar, g1], xp)[:, :2, 0] * d1[:, None]
    a_xy = _funnel(a_xy, p, v, c1, n1, xp, prm, -prm["x_far"], prm["x_end"], True)
    k0 = xp.clip(obs["n_gates_passed"] - 1, 0, obs["gate_sequence"].shape[-1] - 1)
    g0 = obs["gate_sequence"][ar, k0]
    d0 = obs["gate_sequence_direction"][ar, k0]
    c0 = obs["gates_pos"][ar, g0]
    n0 = quat_to_rotmat(obs["gates_quat"][ar, g0], xp)[:, :2, 0] * d0[:, None]
    a_prev = _funnel(a_xy, p, v, c0, n0, xp, prm, -0.05, prm["x_end"], False)
    has_prev = (obs["n_gates_passed"] > 0)[:, None]
    return xp.where(has_prev, a_prev, a_xy)


# Collision-cone CBF (Tayal et al., "Control barrier functions in dynamic UAVs", C3BF):
#   q = o - p (drone -> cylinder), u = -v (relative velocity),  cos(phi) = sqrt(|q|^2 - r^2)/|q|
#   h = q.u + |u| sqrt(|q|^2 - r^2)   >= 0  <=>  the velocity does not point into the collision cone
#   h_dot = c0 - g.a,  g = q + u/|u| sqrt(|q|^2-r^2),  c0 = |u|^2 + |u| (q.u)/sqrt(|q|^2-r^2)
#   constraint h_dot >= -gamma h   ->   g.a <= gamma h + c0   (only acts when heading into the cone)
CONE = dict(gamma=3.0)


def cbf_filter_cone(a_xy, obs, xp, prm=CBF, cone=CONE):
    p, v = obs["pos"], obs["vel"]
    cen, rad, act = _cylinders(obs, xp, prm)
    q_all = cen - p[:, None, :2]
    d = xp.sqrt((q_all**2).sum(-1)) + 1e-6
    order = xp.argsort(-d, axis=-1)
    u = -v[:, :2]
    un = xp.sqrt((u**2).sum(-1)) + 1e-6
    for k in range(cen.shape[1]):
        j = order[:, k]
        q = xp.take_along_axis(q_all, j[:, None, None], 1)[:, 0]
        dj = xp.take_along_axis(d, j[:, None], 1)[:, 0]
        rj = xp.take_along_axis(rad, j[:, None], 1)[:, 0]
        aj = xp.take_along_axis(act, j[:, None], 1)[:, 0]
        s = xp.sqrt(xp.maximum(dj**2 - rj**2, 1e-4))
        qu = (q * u).sum(-1)
        h = qu + un * s
        g = q + u / un[:, None] * s[:, None]
        c0 = un**2 + un * qu / s
        ga = (g * a_xy).sum(-1)
        lim = cone["gamma"] * h + c0
        # degenerate at low relative speed: act only when moving > 0.5 m/s towards a close cylinder
        active = aj & (dj - rj < 1.0) & (un > 0.5) & (qu < 0)
        corr = xp.where(active & (ga > lim), (ga - lim) / ((g**2).sum(-1) + 1e-9), 0.0)
        a_xy = a_xy - corr[:, None] * g
    nrm = xp.sqrt((a_xy**2).sum(-1, keepdims=True)) + 1e-9
    return a_xy * xp.minimum(1.0, prm["a_max"] / nrm)


# ---------------------------------------------------------------------------------------------
# Version 3: v2 + the third gate ahead (longer look-ahead for planning the turns).
# Appended at the end so a v2 policy can be extended with zero input weights (same behaviour).
N_OBS3 = N_OBS2 + 15


def features_v3(obs, last_action, prev_vel, xp, vlim=None):
    x2 = features_v2(obs, last_action, prev_vel, xp, vlim)
    n = obs["pos"].shape[0]
    yaw = frame_yaw(obs, xp)
    c, s = xp.cos(yaw), xp.sin(yaw)
    seq, sdir = obs["gate_sequence"], obs["gate_sequence_direction"]
    n_seq = seq.shape[-1]
    k3 = xp.clip(obs["n_gates_passed"] + 2, 0, n_seq - 1)
    ar = xp.arange(n)
    cw3, nw3 = _gate_feats(obs, seq[ar, k3], sdir[ar, k3], xp)
    c3h = _rot(cw3.reshape(n, 4, 3), c[:, None], s[:, None]).reshape(n, 12)
    n3h = _rot(nw3, c, s)
    return xp.concatenate([x2, c3h, n3h], -1)


# ---------------------------------------------------------------------------------------------
# Version 4: v2 + "already seen" flags (exact pose known) for the next two gates and the two
# nearest poles. Before a gate / pole is seen, its observed pose is the nominal one (error up to
# ~0.15 m / 0.2 rad); the flag tells the policy when it can stop being cautious.
N_OBS4 = N_OBS2 + 4


def features_v4(obs, last_action, prev_vel, xp, vlim=None):
    x2 = features_v2(obs, last_action, prev_vel, xp, vlim)
    n = obs["pos"].shape[0]
    ar = xp.arange(n)
    g1, _, g2, _ = gate_targets(obs, xp)
    gv = obs["gates_visited"]
    dxy = obs["obstacles_pos"][:, :, :2] - obs["pos"][:, None, :2]
    order = xp.argsort(xp.sqrt((dxy**2).sum(-1)), axis=-1)[:, :2]
    ov = xp.take_along_axis(obs["obstacles_visited"], order, axis=1)
    flags = xp.stack([gv[ar, g1], gv[ar, g2], ov[:, 0], ov[:, 1]], -1).astype(x2.dtype)
    return xp.concatenate([x2, flags], -1)


# ---------------------------------------------------------------------------------------------
# Version 5: v2 + the command latency the policy has to expect (in 20 ms control steps / 3).
# Training: randomized post-filter delay 0-60 ms, the observed value carries +-0.5 step error.
# Deployment: set to the latency measured on the real system (0 in the official simulation).
N_OBS5 = N_OBS2 + 1


def features_v5(obs, last_action, prev_vel, xp, vlim=None, delay_steps=None):
    x2 = features_v2(obs, last_action, prev_vel, xp, vlim)
    n = obs["pos"].shape[0]
    d = (xp.zeros((n, 1), dtype=x2.dtype) if delay_steps is None
         else xp.reshape(delay_steps, (-1, 1)).astype(x2.dtype))
    return xp.concatenate([x2, d / 3.0], -1)


# ---------------------------------------------------------------------------------------------
# Version 6: v2 + the nearest gate that is neither the target nor the next one (its frame is an
# obstacle the v2 policy cannot see; ~half of the remaining gate-frame crashes, update log #32).
# Appended at the end so a v2 policy can be extended with zero input weights (same behaviour).
N_OBS6 = N_OBS2 + 15


def features_v6(obs, last_action, prev_vel, xp, vlim=None):
    x2 = features_v2(obs, last_action, prev_vel, xp, vlim)
    n = obs["pos"].shape[0]
    yaw = frame_yaw(obs, xp)
    c, s = xp.cos(yaw), xp.sin(yaw)
    g1, _, g2, _ = gate_targets(obs, xp)
    G = obs["gates_pos"].shape[1]
    dist = xp.sqrt(((obs["gates_pos"] - obs["pos"][:, None, :]) ** 2).sum(-1))  # (n,G)
    ids = xp.arange(G)[None, :]
    dist = xp.where((ids == g1[:, None]) | (ids == g2[:, None]), 1e3, dist)
    g3 = xp.argmin(dist, axis=-1)
    cw3, nw3 = _gate_feats(obs, g3, xp.ones_like(g3), xp)
    c3h = _rot(cw3.reshape(n, 4, 3), c[:, None], s[:, None]).reshape(n, 12)
    n3h = _rot(nw3, c, s)
    return xp.concatenate([x2, c3h, n3h], -1)
