#!/bin/bash
# 第 3 步：RL 控制器慢速试飞（限速 2 m/s），同时记日志
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p practice/dev/latency/logs
RL_VCAP=2.0 RL_LOG=$PWD/practice/dev/latency/logs/rl_slow python scripts/deploy.py --config level3.toml
