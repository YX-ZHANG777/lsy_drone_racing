"""Replay flight logs (race_ppo.py RL_LOG) through DelayEstimator; print the estimate over time.
usage: python est_offline.py 'est_logs/k2_*' [key=value ...]   (DelayEstimator kwargs)"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "lsy_drone_racing_fresh"))
import numpy as np

from lsy_drone_racing.control import delay_pred as DP
from lsy_drone_racing.control.delay_est import ArxDelayEstimator, DelayEstimator

kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
if "gmin" in kw:
    kw["grid"] = np.arange(kw.pop("gmin"), 4.01, 0.5)
if "horizon" in kw:
    kw["horizon"] = int(kw["horizon"])
prm = DP.load_model_params()
for f in sorted(glob.glob(sys.argv[1])):
    d = np.load(f)
    kk = dict(kw)
    arx = kk.pop("arx", 0)
    noise = kk.pop("noise", 0.0)  # std (rad) of Gaussian noise on roll/pitch (measurement noise test)
    rng = np.random.default_rng(0)
    est = ArxDelayEstimator(Path.home() / "lsy_drone_racing_fresh/lsy_drone_racing/control/delay_arx.npz", **kk) if arx else DelayEstimator(prm, **kk)
    out = []
    for k in range(len(d["t"])):
        q = d["quat"][k]
        if noise > 0:
            from scipy.spatial.transform import Rotation as R
            q = (R.from_rotvec(np.r_[rng.normal(0, noise, 2), 0.0]) * R.from_quat(q)).as_quat()
        obs = dict(pos=d["pos"][k], quat=q, vel=d["vel"][k], ang_vel=d["ang_vel"][k])
        out.append(est.update(obs))
        est.sent(d["cmd"][k])
    out = np.array(out)
    sel = [25, 50, 100, 150, 200, 300]
    print(f"{Path(f).name:28s} n={len(out):3d} " + " ".join(f"t{s * 0.02:.1f}s={out[min(s, len(out) - 1)]:.2f}" for s in sel)
          + f"  J={np.round(est.J / est.J.min(), 2)}")
