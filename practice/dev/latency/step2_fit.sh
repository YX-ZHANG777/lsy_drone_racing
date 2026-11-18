#!/bin/bash
# 第 2 步：计算延迟（不飞，只算）。把屏幕拍照发给 Claude
cd "$(dirname "$0")/../../.." || exit 1
python practice/dev/latency/latency_fit.py "$PWD/practice/dev/latency/logs/real_*"
