# 11/18 lab session：测真机的指令延迟

## 现场顺序（速查）
0. 先做文末「在实验室电脑上准备」：进 fork 根目录 → `unzip -o 压缩包` → `source practice/dev/latency/setup_lab.sh`（自动设好 `FORK`、`LAB`，打印版本、改动、无人机 id/channel，测试控制器能加载）。
1. 问助教规则 → 探测飞行（下文「步骤」3），能的话重复 2–3 次。
2. 现场就算一次 `latency_fit.py`，看 `loop period mean` ≈ 20 ms、`R^2` 接近 1。
3. RL 试飞（level3 赛道，限速 + 记日志）：
   ```bash
   RL_VCAP=2.0 RL_LOG=$LAB/logs/rl python scripts/deploy.py --config level3.toml
   ```
4. 平稳的话去掉 `RL_VCAP` 再飞（仍加 `RL_LOG`）。推力明显不够时换 u5（见「真机试飞 u13 的注意事项」）。
5. 把 `$LAB/logs/` 全部拷走。

## 为什么要测
- 仿真里策略对延迟非常敏感（s17，延迟加在安全滤波器之后，GPU 每档 4096 局）：0 ms 97.9%，20 ms 89.7%，40 ms 62.0%，60 ms 11.8%。
- 真机上有仿真里没有的延迟：动捕 → 状态估计 → 电脑 → 无线电 → 无人机。这个数不知道，只能实测。
- 测到以后：① 检查训练时的延迟范围（0–60 ms）是否覆盖真机；② 检查控制器里的**在线延迟估计**（11/16 晚加入，`control/delay_est.py`）在真机上是否给出同样的数；③ 起飞初值 `d_init`（现在 1.5 步 = 30 ms）可以按实测值调整。
- ⚠️ 11/16 晚更新：不再需要手动设 `RL_DELAY_MS`。预测型策略（npz 里 `pred_model=1`）会自己估计延迟。
- ⚠️ 11/17 晚更新：默认控制器现在是 **u13**（`control/race_ppo.npz`，预测型安全屏障 + 姿态滞后预测，倾角上限 1.0）。上面第 4 行的数字是 s17 时期的背景；u13 在仿真里 0–60 ms 延迟下都在 96–98%。

## 原理（一句话）
记录“发出的姿态指令”和“动捕测到的姿态”，算出指令 → 姿态的阶跃响应升到一半的时间 t50。
仿真里同一个测量机动的 t50 = **94.9 ms**，真机多出来的部分就是额外延迟。
仿真验证：人为加 20 / 40 ms 延迟，测出 +19.8 / +39.9 ms。

## 步骤
1. **先问助教**：能不能飞自己的姿态模式控制器、在哪里飞、要注意什么。按实验室安全规则来。
2. 起飞前把无人机放在起点，周围 1 m 内没有门和杆子。
3. 运行（在 fork 根目录 `$FORK` 里，用 fork 自己的 Python 环境）：
   ```bash
   PROBE_LOG=$LAB/logs/real \
   python scripts/deploy.py --config $LAB/probe.toml
   ```
   飞行过程（共约 28 s）：3 s 爬升到起点上方 1 m → 20 s 悬停，同时 roll/pitch 加 ±0.1 rad（约 6°）的小幅随机摆动 → 5 s 降到 0.15 m 高度后结束。
   如果晃得厉害，可以设 `PROBE_AMP=0.05` 把摆动幅度减半。
4. 如果时间允许，重复 2–3 次（结果应该相差几毫秒以内）。
5. 回来以后计算：
   ```bash
   python $LAB/latency_fit.py "$LAB/logs/real_*"
   ```
   看输出里的 `mean t50`，减去 94.9 ms 就是额外延迟。同时检查 `loop period mean` 是否约 20 ms（控制循环有没有跟上）、`R^2` 是否接近 1。
6. **第二种方法交叉检验**（同一份日志）：
   ```bash
   python $LAB/est_offline.py "$LAB/logs/real_*" arx=1 horizon=4 n_min=5 z_min=0.05 d_init=1.5
   ```
   输出是按时间的延迟估计（单位：控制周期，1 = 20 ms）。仿真验证：注入 0 / 1 / 2 步时给出 0.00 / 0.88 / 1.88。两种方法的结果应该差不多（相差 5 ms 以内）；差很多说明真机的姿态响应和仿真不一样，要再分析。

## 额外记录（如果有机会）
- 飞 RL 策略时加 `RL_LOG=$LAB/logs/rl`，日志可以用同一个脚本分析（基准要换成仿真里 u13 比赛飞行的 **107.9 ms**；以前 s33 是 114.8 ms。这个基准和飞法有关，只用来交叉检验，以探测机动的结果为准）。日志里还有 `d_hat`（控制器每步的延迟估计），可以直接看它收敛到多少。
- 第一次飞 RL 策略时建议加 `RL_VCAP=2.0`（滤波器限速，只用于试飞）。
- 问助教：动捕频率、状态估计器类型、无线电的通信频率；往年 Level 3 的仿真 / 真机成绩。

## 真机试飞 u13 的注意事项（11/17 晚加）
- 第一次飞加 `RL_VCAP=2.0`（滤波器限速 2 m/s，只用于试飞）。低速时倾角远用不到 1.0。
- 如果真机推力明显偏弱（悬停推力指令很高、加速时掉高），换回倾角 0.9 的 u5：
  `RL_POLICY=$FORK/practice/dev/snapshots/2026-11-17_u13/race_ppo_u5_before_switch.npz`
- 看 `d_hat`：仿真 0 ms 时 1 s 后平均 0.0–0.1 步；真机上它收敛到的值 × 20 ms 应该和探测机动测出的额外延迟差不多。

## 在实验室电脑上准备（第一件事，不要 commit）
压缩包 `to_lab_1118.zip`（学校电脑 `~/practice/to_lab_1118.zip`）里的路径和 fork 根目录对齐：
`lsy_drone_racing/control/…`、`config/level0–3.toml`、`practice/dev/latency/…`（测延迟工具 + 本说明）、
`practice/dev/snapshots/2026-11-17_u13/…`（u5 备用 npz）。

1. 把 zip 带到实验室电脑（U 盘 / 发给自己 / `scp` 从学校电脑拿）。
2. 进 fork 根目录，先确认工作区干净、记下 fork 版本：
   ```bash
   cd /path/to/your/fork          # 实验室电脑上的 fork
   export FORK=$PWD LAB=$PWD/practice/dev/latency
   git status --short             # 最好是空的
   git log -1 --oneline           # 开发用的上游版本是 c576683 (2026-10-20, "Upgrade estimators #152")
   ```
   fork 比 c576683 旧很多的话，`race_ppo.py` 用到的 `crazyflow.dynamics.load_params` 可能不存在，先看一眼。
3. 解压（会覆盖 `config/level*.toml`，它们在 git 里，随时可用 `git checkout -- config/` 还原）：
   ```bash
   unzip -o /path/to/to_lab_1118.zip -d "$FORK"
   mkdir -p "$LAB/logs"
   git status --short             # 应看到 4 个 config 改动 + control/ 下 7 个新文件 + practice/
   ```
4. **核对无人机编号**：`config/level*.toml` 和 `$LAB/probe.toml` 里写的是 `id = 10`、`channel = 100`。
   实验室分到的无人机不同的话，把这几个文件的 `[[deploy.drones]]` 改成助教给的值。
5. 快速检查能加载（不起飞）：
   ```bash
   python -c "import lsy_drone_racing.control.race_ppo as m; print(m.POLICY)"
   ```
6. 结束后：日志在 `$LAB/logs/`，拷回学校电脑的 `~/practice/dev/latency/logs/`。
   **什么都不要 commit / push**；想还原 fork 就 `git checkout -- config/` 并删掉新文件。

## 实验室电脑是英文界面时
大概率是 Ubuntu（Linux）。只需要用到「终端」：
- 打开终端：`Ctrl+Alt+T`；或者点左上角 **Activities**（或左下角九个点 **Show Applications**），输入 `terminal`，回车。
- 终端里**粘贴是 `Ctrl+Shift+V`**（不是 `Ctrl+V`），复制是 `Ctrl+Shift+C`。`Ctrl+C` 是**中止正在运行的程序**（无人机飞行中按它 = 停止控制，只在助教同意时用）。
- 找 fork 在哪：`ls ~`、`cd ~/某个目录`；`pwd` 显示当前在哪。U 盘一般在 `/media/用户名/U盘名/`。
- 可能看到的英文：`Permission denied`（没权限，问助教）、`No such file or directory`（路径写错）、`command not found`（环境没激活，问助教怎么进 Python 环境，例如 `pixi shell`）、`ModuleNotFoundError`（同上）。

可以直接对助教说的英文：
- "Which computer and which Python environment should I use to run `deploy.py`?"
- "Is it OK to fly my own attitude-mode controller? I'd like to first run a short latency test: 20 s hover with small roll/pitch excitation (±6°)."
- "Which drone ID and radio channel should I use?"
- "What is the motion capture rate and the radio command rate?"
- "Can I copy the log files to a USB stick afterwards?"
