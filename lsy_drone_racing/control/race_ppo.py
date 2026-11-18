"""PPO racing policy (pure numpy inference).

The policy maps gate-relative observations directly to attitude commands
[roll, pitch, yaw, collective thrust]. Training code: practice/dev/rl_race/ (JAX PPO on GPU).
Weights: ``race_ppo.npz`` next to this file, or the path in env var ``RL_POLICY``.
Use with ``env.control_mode = "attitude"``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control import delay_pred as DP
from lsy_drone_racing.control import race_ppo_features as F
from lsy_drone_racing.control.delay_est import ArxDelayEstimator

POLICY = Path(os.environ.get("RL_POLICY", Path(__file__).parent / "race_ppo.npz"))
VLIM = float(os.environ.get("RL_VLIM", "4.0"))  # speed level (m/s) given to the policy
# expected command latency (ms) for delay-conditioned policies (v5); 0 in the official simulation,
# the measured value on the real drone
RL_DELAY_MS = float(os.environ.get("RL_DELAY_MS", "0"))
# optional flight log for latency identification (dev/latency_fit.py): RL_LOG=/path/prefix
RL_LOG = os.environ.get("RL_LOG", "")
_N_INST = [0]  # controller instances created in this process (unique log names)
# latency compensation for policies trained with state prediction (npz flag pred_model):
# the delay is estimated online (delay_est.py); RL_DELAY_FIX=steps fixes it instead (diagnosis only)
RL_DELAY_FIX = os.environ.get("RL_DELAY_FIX", "")
ARX_FILE = Path(__file__).parent / "delay_arx.npz"
USE_CBF = os.environ.get("RL_CBF", "1") == "1"  # pole safety filter (control barrier function)
F.CBF["gate_posts"] = os.environ.get("RL_POSTS", "1") == "1"  # gate side bars as barrier cylinders


class RacePPO(Controller):
    """Neural-network racing policy trained with PPO."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        p = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        self.thrust_min = float(p["thrust_min"] * 4)
        self.thrust_max = float(p["thrust_max"] * 4)
        with np.load(POLICY) as f:
            self.w = {k: f[k].astype(np.float64) for k in f.files}
        self.n_layers = sum(1 for k in self.w if k.startswith("W"))
        self.ver = int(self.w.pop("feat_ver", 1))
        # safety filter parameters stored with the policy (must match training)
        self.use_cbf = USE_CBF and bool(self.w.pop("cbf", 1.0))
        if "cbf_k1" in self.w:
            F.CBF.update(k1=float(self.w.pop("cbf_k1")), k0=float(self.w.pop("cbf_k0")),
                         r_safe=float(self.w.pop("cbf_r_safe")), r_post=float(self.w.pop("cbf_r_post")))
            F.CBF["gate_posts"] = bool(self.w.pop("cbf_posts")) and os.environ.get("RL_POSTS", "1") == "1"
        F.CBF["corridor"] = bool(self.w.pop("cbf_corridor", 0.0))
        F.CBF["corr_slope"] = float(self.w.pop("cbf_corr_slope", 0.0))
        F.MAX_TILT = float(self.w.pop("max_tilt", 0.75))
        # gate top/bottom bars as horizontal barrier cylinders (acts mainly on the thrust)
        F.CBF["bars"] = bool(self.w.pop("cbf_bars", 0.0)) and os.environ.get("RL_BARS", "1") == "1"
        F.CBF["r_bar"] = float(self.w.pop("cbf_r_bar", 0.12))
        F.CBF["bar_wh"] = float(self.w.pop("cbf_bar_wh", 0.1))
        F.CBF["mass"] = float(p["mass"])
        F.CBF["a_max"] = float(self.w.pop("cbf_a_max", 9.0))  # horizontal accel. limit after the filter
        F.CBF["r_unseen"] = float(self.w.pop("cbf_r_unseen", 0.0))
        F.CBF["corr_tau"] = float(self.w.pop("cbf_corr_tau", 0.0))
        F.CBF["pred_T"] = float(self.w.pop("cbf_pred_t", 0.0))  # predictive barrier (0 = classic CBF)
        F.CBF["pred_g"] = float(self.w.pop("cbf_pred_g", 8.0))
        F.CBF["post_seg3"] = bool(self.w.pop("cbf_post3", 0.0))
        F.CBF["bar_pred"] = bool(self.w.pop("cbf_bar_pred", 0.0))
        F.CBF["bar_seg3"] = bool(self.w.pop("cbf_bar3", 0.0))
        F.CBF["wall"] = float(self.w.pop("cbf_wall", 0.0))  # arena-wall barrier margin
        F.CBF["lag_acc"] = float(self.w.pop("cbf_lag", 0.0))  # attitude-lag look-ahead of the barriers
        F.CBF["ground_z"] = float(self.w.pop("cbf_ground_z", 0.0))  # take-off tilt limit
        F.CBF["ground_a"] = float(self.w.pop("cbf_ground_a", 2.0))
        F.CBF["delay"] = float(self.w.pop("cbf_delay", 0.0))
        # delay-conditioned filter: prediction horizon grows with the expected latency
        F.CBF["delay"] += float(self.w.pop("cbf_delay_step", 0.0)) * RL_DELAY_MS / 20.0
        # optional speed level (m/s) enforced by the safety filter, e.g. for first real flights
        F.CBF["v_cap"] = float(os.environ.get("RL_VCAP", "0"))
        self.pred = bool(self.w.pop("pred_model", 0.0))
        if self.pred:
            self.predictor = DP.DeployPredictor(DP.load_model_params(config.sim.drone))
            ek = {k: float(v) for k, v in (x.split("=") for x in os.environ.get("RL_EST", "").split(",") if x)}
            # defaults chosen on the official sim with injected 0-60 ms delays (update log #30); training mimics
            # them (race_env PRED_EST, PRED_D_INIT=1.5, PRED_T_CONV=25)
            ek = {**dict(horizon=4, tau=1.0, n_min=5, z_min=0.05, d_init=1.5), **ek}  # RL_EST="d_init=1.5,z_min=0.05,..." (tuning only)
            ek["horizon"], ek["n_min"] = int(ek["horizon"]), int(ek["n_min"])
            self.est = ArxDelayEstimator(ARX_FILE, **ek)
        self.d_hat = 0.0
        self.last_act = self.w["act_offset"].copy()[None]
        self.prev_vel = np.asarray(obs["vel"], dtype=np.float64)[None]
        self._done = False
        self._log, self._log_ep = [], 0
        _N_INST[0] += 1
        self._log_name = f"{RL_LOG}_{os.getpid()}_{_N_INST[0]}"
        if RL_LOG:
            import atexit
            atexit.register(self._save_log)

    def _mean(self, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - self.w["mean"]) / np.sqrt(self.w["var"] + 1e-8), -10.0, 10.0)
        for i in range(self.n_layers):
            x = x @ self.w[f"W{i}"] + self.w[f"b{i}"]
            if i < self.n_layers - 1:
                x = np.tanh(x)
        return x + self.w["act_offset"]

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return the attitude command [roll, pitch, yaw, thrust]."""
        ob = {k: np.asarray(v)[None] for k, v in obs.items()}
        ob_meas, pv = ob, self.prev_vel
        if self.pred:  # act on the state at the time the new command will take effect
            self.d_hat = float(RL_DELAY_FIX) if RL_DELAY_FIX else self.est.update(obs)
            ob, pv = self.predictor.predict({k: (v.astype(np.float64) if k in ("pos", "vel", "quat", "ang_vel") else v)
                                             for k, v in ob.items()}, pv, self.d_hat)
        if self.ver == 5:
            d_in = self.d_hat if self.pred else RL_DELAY_MS / 20.0  # online estimate when predicting
            x = F.features_v5(ob, self.last_act, pv, np, np.array([VLIM]), np.array([d_in]))
        else:
            feat = {1: F.features, 2: F.features_v2, 3: F.features_v3, 4: F.features_v4, 6: F.features_v6}[self.ver]
            x = feat(ob, self.last_act, pv, np, np.array([VLIM]))
        a = np.clip(self._mean(x), -1.0, 1.0)
        self.last_act = a
        self.prev_vel = ob_meas["vel"].astype(np.float64)
        if self.ver >= 2 and self.use_cbf:
            if F.CBF["lag_acc"] > 0:  # current acceleration from attitude + thrust state
                thr = self.predictor.thr if self.pred else np.array([F.CBF["mass"] * 9.81])
                ob = {**ob, "acc": F.att_acc(ob["quat"], thr, F.CBF["mass"], np)}
            cmd, _ = F.scale_action_v2_safe(a, ob, self.thrust_min, self.thrust_max, np)
        elif self.ver >= 2:
            cmd = F.scale_action_v2(a, F.frame_yaw(ob, np), self.thrust_min, self.thrust_max, np)
        else:
            cmd = F.scale_action(a, self.thrust_min, self.thrust_max, np)
        if self.pred:
            self.predictor.sent(cmd[0], self.d_hat)
            self.est.sent(cmd[0])
        if RL_LOG:
            import time
            o = ob_meas
            self._log.append((time.perf_counter(), o["pos"][0], o["quat"][0], o["vel"][0], o["ang_vel"][0], cmd[0], self.d_hat))
            if len(self._log) % 50 == 0:
                self._save_log()
        return cmd[0].astype(np.float32)

    def _save_log(self):
        """Write the log of this episode: time, measured state and the command sent."""
        if not self._log:
            return
        t, pos, quat, vel, angv, cmd, d_hat = (np.array(v) for v in zip(*self._log))
        np.savez(f"{self._log_name}_ep{self._log_ep}.npz", t=t, pos=pos, quat=quat, vel=vel, ang_vel=angv, cmd=cmd,
                 d_hat=d_hat)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        return self._done

    def episode_callback(self):
        if RL_LOG:
            self._save_log()
            self._log, self._log_ep = [], self._log_ep + 1
        self.last_act = self.w["act_offset"].copy()[None]
        self.prev_vel = None
