#!/usr/bin/env bash
# Qwen-Image server on port ${1:-8091}, restarted whenever it exits
cd "$(dirname "$0")/../../.." || exit 1
while true; do
  HF_HUB_OFFLINE=1 .venv-target-generator/bin/python realm/agent/target_generator/server.py "${1:-8091}" > /dev/null 2>&1
  sleep 5
done
