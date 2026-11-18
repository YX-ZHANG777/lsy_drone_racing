"""Latency measurement flight (attitude mode): hover 1 m above the start with a PID position loop and
add a small pseudo-random roll/pitch excitation (+-0.1 rad, held 3-8 control steps) to the command.
Every step (time, measured state, command) is logged for practice/dev/latency/latency_fit.py.

Phases: 0-3 s climb to the hover point, 3-23 s excitation, 23-28 s descend to 0.15 m, then finished.
Log file prefix: environment variable PROBE_LOG (default ./latency_probe).
Usage (real drone): python scripts/deploy.py --config level3.toml --controller latency_probe.py
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

AMP = float(os.environ.get("PROBE_AMP", "0.1"))  # excitation amplitude (rad)
T_UP, T_EXC, T_DOWN = 3.0, 20.0, 5.0


class LatencyProbe(Controller):
    """Hover + small roll/pitch excitation, logs everything."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self.mass = load_dynamics_params(config.sim.dynamics, config.sim.drone)["mass"]
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)
        self.start = np.asarray(obs["pos"], dtype=float).copy()
        self.hover = self.start + np.array([0.0, 0.0, 1.0])
        self._tick = 0
        self._finished = False
        # pseudo-random excitation: new level every 3-8 steps, independent for roll and pitch
        rng = np.random.default_rng(0)
        n = int((T_UP + T_EXC + T_DOWN) * self._freq) + 10
        self.exc = np.zeros((n, 2))
        for ax in range(2):
            k = int(T_UP * self._freq)
            while k < int((T_UP + T_EXC) * self._freq):
                hold = int(rng.integers(3, 9))
                self.exc[k : k + hold, ax] = AMP * rng.choice([-1.0, 1.0])
                k += hold
        self._log = []
        self._prefix = os.environ.get("PROBE_LOG", "latency_probe") + time.strftime("_%Y%m%d_%H%M%S")

    def _setpoint(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t < T_UP:  # smooth climb
            s = 0.5 - 0.5 * math.cos(math.pi * t / T_UP)
            ds = 0.5 * math.pi / T_UP * math.sin(math.pi * t / T_UP)
            return self.start + s * (self.hover - self.start), ds * (self.hover - self.start)
        if t < T_UP + T_EXC:
            return self.hover, np.zeros(3)
        low = self.start + np.array([0.0, 0.0, 0.15])
        s = min((t - T_UP - T_EXC) / T_DOWN, 1.0)
        return self.hover + s * (low - self.hover), (low - self.hover) / T_DOWN * (s < 1.0)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        t = self._tick / self._freq
        if t >= T_UP + T_EXC + T_DOWN:
            self._finished = True
        des_pos, des_vel = self._setpoint(t)
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]
        self.i_error = np.clip(self.i_error + pos_error / self._freq, -self.ki_range, self.ki_range)
        f = self.kp * pos_error + self.ki * self.i_error + self.kd * vel_error
        f[2] += self.mass * 9.81
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust = f.dot(z_axis)
        z_des = f / np.linalg.norm(f)
        x_c = np.array([1.0, 0.0, 0.0])  # yaw 0
        y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)
        eul = R.from_matrix(np.vstack([x_des, y_des, z_des]).T).as_euler("xyz")
        eul[:2] += self.exc[min(self._tick, len(self.exc) - 1)]
        cmd = np.concatenate([eul, [thrust]]).astype(np.float32)
        self._log.append((time.perf_counter(), np.asarray(obs["pos"]), np.asarray(obs["quat"]),
                          np.asarray(obs["vel"]), np.asarray(obs["ang_vel"]), cmd))
        if len(self._log) % 50 == 0:
            self._save()
        return cmd

    def _save(self):
        if self._log:
            t, pos, quat, vel, angv, cmd = (np.array(v) for v in zip(*self._log))
            np.savez(self._prefix + ".npz", t=t, pos=pos, quat=quat, vel=vel, ang_vel=angv, cmd=cmd)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._save()
