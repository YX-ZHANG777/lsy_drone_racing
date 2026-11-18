"""Estimate the command-to-attitude delay from flight logs (race_ppo.py with RL_LOG=...).

Model-free: regress the measured roll/pitch on the past commanded roll/pitch (FIR, 0.6 s long),
integrate the impulse response to a step response and report the time it needs to reach 50 % (t50).
Mocap + estimator latency and radio latency both shift this curve, so
    extra delay of the real system = t50(real logs) - t50(simulation logs without delay).

usage: python latency_fit.py LOG_GLOB [LOG_GLOB ...]      (each glob is evaluated separately)
"""
import glob
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

DT = 0.02  # control period (50 Hz)
J = 30  # FIR length (steps)


def load(pattern):
    eps = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        if len(d["t"]) < J + 20:
            continue
        eul = R.from_quat(d["quat"]).as_euler("xyz")  # measured roll, pitch, yaw
        eps.append(dict(t=d["t"], y=eul[:, :2], u=d["cmd"][:, :2]))
    return eps


def fit(eps, axis):
    """Least-squares FIR y[k] = c + sum_j h_j u[k-j]; returns h (J+1,)."""
    X, Y = [], []
    for e in eps:
        u, y = e["u"][:, axis], e["y"][:, axis]
        n = len(u)
        # log entry k: state measured BEFORE command k was sent -> command k cannot act on y[k]
        for k in range(J + 1, n):
            X.append(np.r_[1.0, u[k - np.arange(J + 1)]])  # u[k], u[k-1], ..., u[k-J]
            Y.append(y[k])
    X, Y = np.array(X), np.array(Y)
    lam = 1e-4 * len(Y)
    A = X.T @ X + lam * np.diag(np.r_[0.0, np.ones(X.shape[1] - 1)])
    th = np.linalg.solve(A, X.T @ Y)
    res = Y - X @ th
    return th[1:], 1.0 - res.var() / Y.var()


def t50(h):
    s = np.cumsum(h)
    s = s / s[-1]
    i = int(np.argmax(s >= 0.5))
    # linear interpolation between steps i-1 and i (lag j means command j steps before the sample)
    f = (0.5 - s[i - 1]) / (s[i] - s[i - 1]) if i > 0 else 0.0
    return (i - 1 + f) * DT, s


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        eps = load(pat)
        if not eps:
            print(f"{pat}: no logs")
            continue
        dts = np.concatenate([np.diff(e["t"]) for e in eps])
        out = []
        for ax, name in ((0, "roll"), (1, "pitch")):
            h, r2 = fit(eps, ax)
            t, s = t50(h)
            out.append(t)
            print(f"{pat}: {name:5s} t50 = {t * 1000:5.1f} ms  gain = {h.sum():.3f}  R^2 = {r2:.4f}  "
                  f"step response (first 6 steps) = {np.round(s[:6], 2)}")
        print(f"{pat}: {len(eps)} episodes, {sum(len(e['t']) for e in eps)} samples, "
              f"loop period mean {dts.mean() * 1000:.1f} ms (std {dts.std() * 1000:.1f}), "
              f"mean t50 = {np.mean(out) * 1000:.1f} ms")
