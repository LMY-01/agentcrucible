#!/usr/bin/env bash
# kb-agent 垂直层：复用 arlab 运行时 + kb-lab 的检索索引。
#   ./run.sh ask "LoRA 的核心思想是什么？"
#   ./run.sh eval --limit 3
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_ROOT="$(cd ../.. && pwd)"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$HOME/venvs/kb-lab}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
# src 给运行时，PROJECT_ROOT 给 apps 包，kb-lab 的 src 给检索
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${KB_LAB_ROOT:-/mnt/d/kb-lab}/src${PYTHONPATH:+:$PYTHONPATH}"
export KB_LAB_ROOT="${KB_LAB_ROOT:-/mnt/d/kb-lab}"

# 关键：sentence_transformers 默认会在加载权重前向 HuggingFace Hub 查 revision。
# WSL 里这个请求不可达，会阻塞到 TCP 超时（实测 >7 分钟），表现为「程序卡死」。
# 模型权重本地已缓存，强制离线即可，顺带把首跳从几十秒压到 2 秒。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [ -z "${ARLAB_API_KEY:-}" ] && [ -f "$HOME/.hermes/.arlab-key" ]; then
  export ARLAB_API_KEY="$(cat "$HOME/.hermes/.arlab-key")"
fi

# WSL 里代理变量会劫持到本机端口，连 Windows 主机网关时全部失败
unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

cmd="${1:-eval}"
shift || true
exec python -m apps.kb_agent.app "$cmd" "$@"
