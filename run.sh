#!/usr/bin/env bash
# 一键进入项目环境并跑命令。用法：
#   ./run.sh eval           跑全部评测
#   ./run.sh eval --limit 2 只跑前两条
#   ./run.sh run "任务"      跑单个任务
set -euo pipefail
cd "$(dirname "$0")"

export VIRTUAL_ENV="${VIRTUAL_ENV:-$HOME/venvs/arlab}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 网关密钥走环境变量，不写进仓库
if [ -z "${ARLAB_API_KEY:-}" ] && [ -f "$HOME/.hermes/.arlab-key" ]; then
  export ARLAB_API_KEY="$(cat "$HOME/.hermes/.arlab-key")"
fi

# WSL 里 ALL_PROXY 会劫持到本机代理端口，导致连 Windows 主机 IP 失败
unset ALL_PROXY all_proxy http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

cmd="${1:-eval}"
shift || true
exec python -m arlab "$cmd" "$@"
