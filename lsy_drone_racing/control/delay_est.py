"""Online estimation of the command latency (multiple-model comparison).

For each candidate delay d (a grid in control periods, fractional values allowed) the identified
model (``delay_pred``) is integrated from the state measured ``horizon`` periods ago, using the
commands that would have been executed if the delay were d. The predicted attitude is compared with
the measured one; squared errors are accumulated with exponential forgetting and the candidate
with the smallest accumulated error (refined by a parabola through its neighbours) is the estimate.

Only quantities the controller knows are used (its own commands and the measured states), so the
same code runs unchanged in simulation and on the real drone.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from lsy_drone_racing.control import delay_pred as DP


class DelayEstimator:
    """Estimate the delay (in control periods) between sending a command and its execution."""

    def __init__(self, prm: dict, grid=None, horizon: int = 3, tau: float = 1.0, d_init: float = 0.0,
                 w_rate: float = 0.02, z_min: float = 0.1, thr0: float | None = None):
        """Args:
        prm: Model parameters (``delay_pred.load_model_params``).
        grid: Candidate delays in control periods.
        horizon: Prediction length in control periods for each comparison.
        tau: Forgetting time constant (s) of the accumulated errors.
        d_init: Estimate returned before enough data has been collected.
        w_rate: Weight of the roll/pitch-rate error (rad/s) relative to the angle error (rad).
        z_min: Updates only above this height (ground contact is not in the model).
        thr0: Initial thrust state (N); default hover thrust.
        """
        self.prm = prm
        self.grid = np.arange(0.0, 4.01, 0.5) if grid is None else np.asarray(grid, float)
        self.m = int(horizon)
        self.lam = math.exp(-DP.DT / tau)
        self.w_rate = w_rate
        self.z_min = z_min
        self.d_init = d_init
        self.thr0 = prm["mass"] * prm["g"] if thr0 is None else thr0
        self.reset()

    def reset(self):
        self.cmds = []  # all commands sent this episode (index = control step)
        self.states = deque(maxlen=self.m + 1)  # (step, euler state tuple) of the last m+1 steps
        self.thr_hist = deque(maxlen=self.m + 1)  # per-hypothesis thrust state at those steps
        self.thr = np.full(len(self.grid), float(self.thr0))
        self.J = np.zeros(len(self.grid))
        self.n_upd = 0
        self.t = 0
        self.d_hat = self.d_init

    def _executed(self, t0: int, n_periods: int) -> np.ndarray:
        """(G, n_periods * N_SUB, 4) commands executed from step t0 on, for every candidate delay."""
        S = DP.N_SUB
        tau = t0 + np.arange(n_periods * S) / S  # time in control periods
        idx = np.floor(tau[None, :] - self.grid[:, None] + 1e-9).astype(int)
        idx = np.clip(idx, 0, len(self.cmds) - 1)
        return np.asarray(self.cmds)[idx]

    def update(self, obs: dict) -> float:
        """Call once per control step with the new measurement, BEFORE sending the next command.

        Returns the current delay estimate in control periods.
        """
        st = DP.euler_state({k: np.asarray(obs[k], float)[None] for k in ("pos", "vel", "quat", "ang_vel")}, np)
        if self.t > 0 and self.cmds:
            # advance the per-hypothesis thrust states by one period
            G = len(self.grid)
            ex = self._executed(self.t - 1, 1)
            decay = math.exp(-self.prm["k_thr"] * DP.DT / DP.N_SUB)
            for i in range(ex.shape[1]):
                self.thr = ex[:, i, 3] + (self.thr - ex[:, i, 3]) * decay
            if len(self.states) == self.m + 1 and float(st[0][0, 2]) > self.z_min:
                t0, st0 = self.states[0]
                thr0 = self.thr_hist[0]
                rep = tuple(np.repeat(v, G, axis=0) for v in st0)
                pred, _ = DP.integrate_sub(rep, thr0.copy(), self._executed(t0, self.m), self.prm, np)
                e_ang = (pred[2] - st[2]) ** 2 + (pred[3] - st[3]) ** 2
                e_rate = (pred[5] - st[5]) ** 2 + (pred[6] - st[6]) ** 2
                self.J = self.lam * self.J + e_ang + self.w_rate * e_rate
                self.n_upd += 1
                self.d_hat = self._argmin()
        self.states.append((self.t, st))
        self.thr_hist.append(self.thr.copy())
        return self.d_hat

    def _argmin(self) -> float:
        if self.n_upd < 10:
            return self.d_init
        i = int(np.argmin(self.J))
        if 0 < i < len(self.grid) - 1:  # parabola through the neighbours
            a, b, c = self.J[i - 1], self.J[i], self.J[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den > 1e-12 else 0.0
            return float(self.grid[i] + np.clip(off, -0.5, 0.5) * (self.grid[1] - self.grid[0]))
        return float(self.grid[i])

    def sent(self, cmd) -> None:
        """Call with the command that was just sent."""
        self.cmds.append(np.asarray(cmd, float).copy())
        self.t += 1


class ArxDelayEstimator:
    """Delay estimation with an ARX model of the attitude response identified from delay-free
    simulation logs (practice/dev/latency/est_arx.py), so the estimate is the EXTRA delay relative to
    the simulation (0 in the official simulation).

    For candidate delay d = n + f (n integer, 0 <= f < 1) the command acting in period k is
    (1 - f) u[k - n] + f u[k - n - 1]. Each step the measured roll/pitch is compared with the
    ``horizon``-step prediction from the measurements before; squared errors are accumulated with
    exponential forgetting and the best candidate (parabola-refined) is the estimate.
    """

    def __init__(self, model_file, grid=None, horizon: int = 2, tau: float = 1.0, d_init: float = 0.0,
                 z_min: float = 0.1, n_min: int = 10):
        with np.load(model_file) as f:
            self.th = f["theta"].astype(float)
            self.na, self.nb = int(f["na"]), int(f["nb"])
        self.grid = np.arange(0.0, 4.01, 0.5) if grid is None else np.asarray(grid, float)
        self.m = int(horizon)
        self.lam = math.exp(-DP.DT / tau)
        self.d_init, self.z_min, self.n_min = d_init, z_min, n_min
        n = np.floor(self.grid + 1e-9).astype(int)
        self.n0, self.f = n, self.grid - n
        self.reset()

    def reset(self):
        self.y, self.u, self.ok = [], [], []
        self.J = np.zeros(len(self.grid))
        self.n_upd = 0
        self.d_hat = self.d_init

    def _u_shift(self, s: int) -> np.ndarray:
        """(G, 2) command acting in period s -> s+1 for every candidate (index < 0: first command)."""
        u = self.u
        i0 = np.clip(s - self.n0, 0, len(u) - 1)
        i1 = np.clip(s - self.n0 - 1, 0, len(u) - 1)
        U = np.asarray(u)
        return (1 - self.f)[:, None] * U[i0] + self.f[:, None] * U[i1]

    def update(self, obs: dict) -> float:
        """Call once per control step with the new measurement, BEFORE sending the next command."""
        q = np.asarray(obs["quat"], float)[None]
        r, p, _ = DP._quat2rpy(q, np)
        self.y.append(np.array([r[0], p[0]]))
        self.ok.append(float(np.asarray(obs["pos"])[2]) > self.z_min)
        k = len(self.y) - 1
        na, nb, m = self.na, self.nb, self.m
        L = max(na, nb + int(self.grid.max()) + 1) + m
        if k >= L and all(self.ok[k - na - m:]):  # measured states used must be airborne
            G = len(self.grid)
            Y = np.asarray(self.y)
            hist = [np.repeat(Y[k - m - na + i][None], G, 0) for i in range(na)]  # oldest first
            a, b, c = self.th[:, :na], self.th[:, na:na + nb], self.th[:, -1]
            for s in range(k - m, k + 1):
                yp = c[None].copy()
                for i in range(na):
                    yp = yp + a[None, :, i] * hist[-1 - i]
                for j in range(nb):
                    yp = yp + b[None, :, j] * self._u_shift(s - 1 - j)
                hist.append(yp)
            e = np.sum((hist[-1] - Y[k][None]) ** 2, -1)
            self.J = self.lam * self.J + e
            self.n_upd += 1
            if self.n_upd >= self.n_min:
                self.d_hat = self._argmin()
        return self.d_hat

    def _argmin(self) -> float:
        i = int(np.argmin(self.J))
        if 0 < i < len(self.grid) - 1:
            a, b, c = self.J[i - 1], self.J[i], self.J[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den > 1e-15 else 0.0
            return float(self.grid[i] + np.clip(off, -0.5, 0.5) * (self.grid[1] - self.grid[0]))
        return float(self.grid[i])

    def sent(self, cmd) -> None:
        """Call with the command that was just sent."""
        self.u.append(np.asarray(cmd, float)[:2].copy())
