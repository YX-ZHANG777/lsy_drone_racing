#!/bin/bash
# 第 1 步：测延迟的飞行（约 28 秒：起飞 → 悬停小幅摆动 → 降低）
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p practice/dev/latency/logs
PROBE_LOG=$PWD/practice/dev/latency/logs/real python scripts/deploy.py --config $PWD/practice/dev/latency/probe.toml
