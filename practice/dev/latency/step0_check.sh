#!/bin/bash
# 第 0 步：检查。在 fork 文件夹里运行：bash step0_check.sh
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p practice/dev/latency/logs
echo "======== 现在在这个文件夹：$PWD"
echo "======== 无人机编号（要和助教给的一样）："
grep -A2 "deploy.drones" config/level3.toml | grep -E "id =|channel ="
echo "======== 测试控制器能不能加载："
python -c "import lsy_drone_racing.control.race_ppo as m; print('OK 没问题')" || echo "!!!!!!!! 出错了：把这个屏幕拍照，问助教 Python 环境怎么进"
