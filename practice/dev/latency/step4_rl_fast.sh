#!/bin/bash
# 第 4 步：RL 控制器正常速度飞（第 3 步飞得稳才做），同时记日志
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p practice/dev/latency/logs
RL_LOG=$PWD/practice/dev/latency/logs/rl_fast python scripts/deploy.py --config level3.toml
