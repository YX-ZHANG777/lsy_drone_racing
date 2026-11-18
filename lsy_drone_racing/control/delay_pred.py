"""Delay compensation by state prediction (Smith-predictor idea).

The commands already sent but not yet executed are known. Integrating an identified model with them
gives the state at the moment the next command takes effect; the policy and the safety filter act on
that predicted state, so for them the loop looks (almost) delay-free.

Model: crazyflow ``so_rpy_rotor_drag`` (fitted second-order roll/pitch/yaw response, first-order
thrust lag, linear body drag), re-implemented here so that it runs with numpy and jax alike.
Works on batches: obs arrays have a leading dimension n.
"""

from __future__ import annotations

import math

DT = 0.02  # control period (s)
N_SUB = 10  # integration sub-steps per control period (2 ms)


def load_model_params(drone: str = "cf21B_500") -> dict:
    """Identified parameters of so_rpy_rotor_drag as plain floats / lists."""
    from crazyflow.dynamics import load_params

    p = load_params("so_rpy_rotor_drag", drone)
    f = lambda v: [float(x) for x in list(v)]  # noqa: E731
    return dict(
        mass=float(p["mass"]),
        g=-float(list(p["gravity_vec"])[2]),
        k_thr=float(p["thrust_dyn_coef"]),
        acc_coef=float(p["acc_coef"]),
        cmd_f_coef=float(p["cmd_f_coef"]),
        rpy_coef=f(p["rpy_coef"]),
        rpy_rates_coef=f(p["rpy_rates_coef"]),
        cmd_rpy_coef=f(p["cmd_rpy_coef"]),
        drag=[float(p["drag_matrix"][i][i]) for i in range(3)],
    )


def _quat2rpy(q, xp):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = xp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = xp.arcsin(xp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yw = xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return r, p, yw


def _rpy2quat(r, p, y, xp):
    cr, sr = xp.cos(r / 2), xp.sin(r / 2)
    cp, sp = xp.cos(p / 2), xp.sin(p / 2)
    cy, sy = xp.cos(y / 2), xp.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return xp.stack([x, yy, z, w], -1)


def _rot(r, p, y, xp):
    """World-from-body rotation matrix for extrinsic xyz Euler angles, shape (..., 3, 3)."""
    cr, sr, cp, sp, cy, sy = xp.cos(r), xp.sin(r), xp.cos(p), xp.sin(p), xp.cos(y), xp.sin(y)
    row0 = xp.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1)
    row1 = xp.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1)
    row2 = xp.stack([-sp, cp * sr, cp * cr], -1)
    return xp.stack([row0, row1, row2], -2)


def thrust_update(thr, cmd_thrust, prm):
    """Thrust state after one control period with the executed command (exact first-order lag)."""
    a = math.exp(-prm["k_thr"] * DT)
    return cmd_thrust + (thr - cmd_thrust) * a


def predict(obs: dict, cmds, thr, prm: dict, xp, active=None, prev_vel=None):
    """Predict the state after applying ``cmds`` (n, k, 4) one control period each, in order.

    Args:
        obs: Current observation (pos, quat, vel, ang_vel with leading batch dimension).
        cmds: Commands [roll, pitch, yaw, thrust_N] still to be executed, oldest first.
        thr: Current thrust state (n,) in commanded-thrust units.
        prm: Model parameters from ``load_model_params``.
        xp: numpy or jax.numpy.
        active: Optional (n, k) bool mask; step j is only applied where True (per-world delays:
            a world with delay d uses the last d commands, i.e. active[:, j] = j >= k - d).
        prev_vel: Velocity one control period before now (returned unchanged where nothing is applied).

    Returns:
        (predicted obs dict, velocity one control period before the end of the prediction)
    """
    pos, vel = obs["pos"], obs["vel"]
    r, p, y = _quat2rpy(obs["quat"], xp)
    w = obs["ang_vel"]  # body rates -> Euler angle rates
    t = xp.tan(p)
    dr = w[..., 0] + xp.sin(r) * t * w[..., 1] + xp.cos(r) * t * w[..., 2]
    dp = xp.cos(r) * w[..., 1] - xp.sin(r) * w[..., 2]
    dy = (xp.sin(r) * w[..., 1] + xp.cos(r) * w[..., 2]) / xp.cos(p)
    a_r, a_p, a_y = prm["rpy_coef"]
    b_r, b_p, b_y = prm["rpy_rates_coef"]
    c_r, c_p, c_y = prm["cmd_rpy_coef"]
    drag = xp.asarray(prm["drag"], dtype=pos.dtype)
    m, g, h = prm["mass"], prm["g"], DT / N_SUB
    decay = math.exp(-prm["k_thr"] * h)
    prev_v = vel if prev_vel is None else prev_vel
    for j in range(cmds.shape[1]):
        st0 = (pos, vel, r, p, y, dr, dp, dy, thr)
        prev_j = vel
        c = cmds[:, j]
        for _ in range(N_SUB):
            Rw = _rot(r, p, y, xp)
            f = prm["acc_coef"] + prm["cmd_f_coef"] * thr
            vb = xp.einsum("nji,nj->ni", Rw, vel)  # body-frame velocity
            acc = (f / m)[:, None] * Rw[..., :, 2] + xp.einsum("nij,nj->ni", Rw, drag * vb) / m
            acc = acc - xp.asarray([0.0, 0.0, g], dtype=pos.dtype)
            ddr = a_r * r + b_r * dr + c_r * c[:, 0]
            ddp = a_p * p + b_p * dp + c_p * c[:, 1]
            ddy = a_y * y + b_y * dy + c_y * c[:, 2]
            pos = pos + h * vel + 0.5 * h * h * acc
            vel = vel + h * acc
            r, p, y = r + h * dr, p + h * dp, y + h * dy
            dr, dp, dy = dr + h * ddr, dp + h * ddp, dy + h * ddy
            thr = c[:, 3] + (thr - c[:, 3]) * decay
        if active is None:
            prev_v = prev_j
        else:  # keep the old state where this step does not apply
            act = active[:, j]
            prev_v = xp.where(act[:, None], prev_j, prev_v)
            new = (pos, vel, r, p, y, dr, dp, dy, thr)
            pos, vel, r, p, y, dr, dp, dy, thr = (
                xp.where(act[:, None] if a.ndim == 2 else act, a, b) for a, b in zip(new, st0)
            )
    # Euler angle rates -> body rates
    wx = dr - xp.sin(p) * dy
    wy = xp.cos(r) * dp + xp.sin(r) * xp.cos(p) * dy
    wz = -xp.sin(r) * dp + xp.cos(r) * xp.cos(p) * dy
    out = dict(obs)
    out.update(pos=pos, vel=vel, quat=_rpy2quat(r, p, y, xp), ang_vel=xp.stack([wx, wy, wz], -1))
    return out, prev_v


def euler_state(obs: dict, xp):
    """(pos, vel, r, p, y, dr, dp, dy) from an observation (batched)."""
    r, p, y = _quat2rpy(obs["quat"], xp)
    w = obs["ang_vel"]
    t = xp.tan(p)
    dr = w[..., 0] + xp.sin(r) * t * w[..., 1] + xp.cos(r) * t * w[..., 2]
    dp = xp.cos(r) * w[..., 1] - xp.sin(r) * w[..., 2]
    dy = (xp.sin(r) * w[..., 1] + xp.cos(r) * w[..., 2]) / xp.cos(p)
    return obs["pos"], obs["vel"], r, p, y, dr, dp, dy


def integrate_sub(st, thr, cmd_sub, prm: dict, xp):
    """Integrate the model with one command per 2 ms sub-step.

    Args:
        st: (pos, vel, r, p, y, dr, dp, dy) with batch dimension n (see ``euler_state``).
        thr: Thrust state (n,).
        cmd_sub: (n, S, 4) executed command for each of the S sub-steps (S = N_SUB per period),
            so that fractional delays (e.g. 30 ms) can be represented.

    Returns:
        (state tuple, thrust state) after S sub-steps.
    """
    pos, vel, r, p, y, dr, dp, dy = st
    a_r, a_p, a_y = prm["rpy_coef"]
    b_r, b_p, b_y = prm["rpy_rates_coef"]
    c_r, c_p, c_y = prm["cmd_rpy_coef"]
    drag = xp.asarray(prm["drag"], dtype=pos.dtype)
    m, g, h = prm["mass"], prm["g"], DT / N_SUB
    decay = math.exp(-prm["k_thr"] * h)
    for i in range(cmd_sub.shape[1]):
        c = cmd_sub[:, i]
        Rw = _rot(r, p, y, xp)
        f = prm["acc_coef"] + prm["cmd_f_coef"] * thr
        vb = xp.einsum("nji,nj->ni", Rw, vel)
        acc = (f / m)[:, None] * Rw[..., :, 2] + xp.einsum("nij,nj->ni", Rw, drag * vb) / m
        acc = acc - xp.asarray([0.0, 0.0, g], dtype=pos.dtype)
        ddr = a_r * r + b_r * dr + c_r * c[:, 0]
        ddp = a_p * p + b_p * dp + c_p * c[:, 1]
        ddy = a_y * y + b_y * dy + c_y * c[:, 2]
        pos = pos + h * vel + 0.5 * h * h * acc
        vel = vel + h * acc
        r, p, y = r + h * dr, p + h * dp, y + h * dy
        dr, dp, dy = dr + h * ddr, dp + h * ddp, dy + h * ddy
        thr = c[:, 3] + (thr - c[:, 3]) * decay
    return (pos, vel, r, p, y, dr, dp, dy), thr


def state_to_obs(st, obs: dict, xp):
    """Observation dict with pos / vel / quat / ang_vel replaced by the Euler state ``st``."""
    pos, vel, r, p, y, dr, dp, dy = st
    wx = dr - xp.sin(p) * dy
    wy = xp.cos(r) * dp + xp.sin(r) * xp.cos(p) * dy
    wz = -xp.sin(r) * dp + xp.cos(r) * xp.cos(p) * dy
    out = dict(obs)
    out.update(pos=pos, vel=vel, quat=_rpy2quat(r, p, y, xp), ang_vel=xp.stack([wx, wy, wz], -1))
    return out


class DeployPredictor:
    """Single-drone (numpy) state prediction for deployment, fractional delays allowed.

    Keeps the commands already sent and the thrust state; ``predict`` returns the state at the time a
    command sent now starts to act, given the delay ``d`` in control periods (d = 0: measured state).
    For integer d this is the same computation as ``predict`` used in training.
    """

    def __init__(self, prm: dict, n_hist: int = 8):
        import numpy as np

        self.np, self.prm, self.n_hist = np, prm, n_hist
        self.hover = np.array([0.0, 0.0, 0.0, prm["mass"] * prm["g"]])
        self.reset()

    def reset(self):
        np = self.np
        self.hist = [self.hover.copy() for _ in range(self.n_hist)]  # oldest first, newest last
        self.thr = np.array([self.prm["mass"] * self.prm["g"]])

    def _executed(self, start: int, d: float, n_sub: int):
        """Commands for sub-steps i = 0..n_sub-1 at time (newest send step + start) + i / N_SUB."""
        np = self.np
        off = start + np.floor(np.arange(n_sub) / N_SUB - d + 1e-9).astype(int)  # <= 0
        idx = np.clip(len(self.hist) - 1 + off, 0, len(self.hist) - 1)
        return np.stack([self.hist[i] for i in idx])[None]

    def sent(self, cmd, d: float) -> None:
        """Register the command just sent and advance the thrust state over the coming period."""
        np = self.np
        self.hist = self.hist[1:] + [np.asarray(cmd, float).copy()]
        ex = self._executed(0, d, N_SUB)[0]
        decay = math.exp(-self.prm["k_thr"] * DT / N_SUB)
        for c in ex[:, 3]:
            self.thr = c + (self.thr - c) * decay

    def predict(self, obs: dict, prev_vel, d: float):
        """(predicted obs dict, velocity one period before the predicted time); batch dim 1."""
        np = self.np
        n_sub = int(round(d * N_SUB))
        if n_sub <= 0:
            return obs, prev_vel
        ex = self._executed(1, d, n_sub)
        st = euler_state(obs, np)
        thr = self.thr.copy()
        n1 = n_sub - N_SUB  # sub-step index of (prediction end - one period)
        if n1 > 0:
            st1, thr = integrate_sub(st, thr, ex[:, :n1], self.prm, np)
            pv = st1[1]
            st, thr = integrate_sub(st1, thr, ex[:, n1:], self.prm, np)
        else:  # d < 1: interpolate between the previous and the current measured velocity
            pv = obs["vel"] if prev_vel is None else prev_vel + (n_sub / N_SUB) * (obs["vel"] - prev_vel)
            st, thr = integrate_sub(st, thr, ex, self.prm, np)
        return state_to_obs(st, obs, np), pv


def predict_frac(obs: dict, cmd_buf, thr, d, prm: dict, xp, prev_vel):
    """Batched prediction with a per-world fractional delay (training; same result as DeployPredictor).

    Args:
        obs: Measured observation, batch n.
        cmd_buf: (n, K, 4) commands already sent, oldest first, newest last.
        thr: (n,) thrust state.
        d: (n,) delay in control periods (>= 0; at most K is represented exactly).
        prev_vel: (n, 3) velocity measured one period before now.

    Returns:
        (predicted obs, velocity one period before the predicted time)
    """
    n, K = cmd_buf.shape[0], cmd_buf.shape[1]
    S = (K + 1) * N_SUB
    n_sub = xp.round(d * N_SUB).astype(xp.int32)  # (n,)
    i = xp.arange(S)
    off = 1 + xp.floor(i[None, :] / N_SUB - d[:, None] + 1e-9).astype(xp.int32)  # (n, S) <= 0 when active
    idx = xp.clip(K - 1 + off, 0, K - 1)
    ex = xp.take_along_axis(cmd_buf, idx[..., None], axis=1)  # (n, S, 4)
    st = euler_state(obs, xp)
    pv = prev_vel + (xp.minimum(n_sub, N_SUB) / N_SUB)[:, None] * (obs["vel"] - prev_vel)  # used if d < 1
    for j in range(S):
        act = j < n_sub  # (n,)
        if j > 0:  # velocity one period before the end: state after (n_sub - N_SUB) sub-steps
            pv = xp.where(((n_sub - N_SUB) == j)[:, None], st[1], pv)
        new, thr_new = integrate_sub(st, thr, ex[:, j:j + 1], prm, xp)
        st = tuple(xp.where(act[:, None] if a.ndim == 2 else act, a, b) for a, b in zip(new, st))
        thr = xp.where(act, thr_new, thr)
    return state_to_obs(st, obs, xp), pv
