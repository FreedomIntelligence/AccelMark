#!/usr/bin/env bash
# AccelMark — Isolated one-command environment setup for any serving recipe runner.
#
# Usage:
#   bash scripts/setup_env.sh <runner_id> [--serve] [--mirror] [--venv <dir>]
#
# Examples:
#   bash scripts/setup_env.sh nvidia_vllm_47f5d58e --serve
#   bash scripts/setup_env.sh nvidia_sglang_c43a8309 --serve --mirror

set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: bash scripts/setup_env.sh <runner_id> [--serve] [--mirror] [--venv <dir>]"
  echo ""
  echo "Options:"
  echo "  --serve        Also install AccelMark OpenAI-compatible serve dependencies"
  echo "  --mirror       Use Tsinghua PyPI mirror (fast in mainland China)"
  echo "  --venv <dir>   Custom virtual environment path (default: .venv-<runner_id>)"
  exit 0
fi

RUNNER_ID="$1"
shift

INSTALL_SERVE=0
USE_MIRROR=0
VENV_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve)  INSTALL_SERVE=1; shift ;;
    --mirror) USE_MIRROR=1; shift ;;
    --venv)   VENV_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER_DIR="${REPO_ROOT}/runners/${RUNNER_ID}"

if [[ ! -d "${RUNNER_DIR}" ]]; then
  echo "Error: runner '${RUNNER_ID}' not found under ${REPO_ROOT}/runners/" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-${RUNNER_ID}}"
echo "==> Setting up isolated environment for ${RUNNER_ID}"
echo "    venv: ${VENV_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv venv "${VENV_DIR}"
  PIP_CMD=(uv pip install --python "${VENV_DIR}/bin/python")
else
  python3 -m venv "${VENV_DIR}"
  PIP_CMD=("${VENV_DIR}/bin/pip" install)
fi

PIP_EXTRA=()
if [[ "${USE_MIRROR}" -eq 1 ]]; then
  PIP_EXTRA+=(-i https://pypi.tuna.tsinghua.edu.cn/simple)
  echo "    mirror: Tsinghua PyPI"
fi

echo "==> Installing AccelMark core..."
"${PIP_CMD[@]}" "${PIP_EXTRA[@]}" -e "${REPO_ROOT}"

if [[ -f "${RUNNER_DIR}/requirements.txt" ]]; then
  echo "==> Installing runner dependencies (${RUNNER_ID}/requirements.txt)..."
  "${PIP_CMD[@]}" "${PIP_EXTRA[@]}" -r "${RUNNER_DIR}/requirements.txt"
fi

if [[ "${INSTALL_SERVE}" -eq 1 && -f "${REPO_ROOT}/serve/requirements.txt" ]]; then
  echo "==> Installing AccelMark Serve dependencies (OpenAI-compatible API)..."
  "${PIP_CMD[@]}" "${PIP_EXTRA[@]}" -r "${REPO_ROOT}/serve/requirements.txt"
fi

if [[ -f "${RUNNER_DIR}/apply_patches.sh" ]]; then
  echo "==> Applying runner patches (${RUNNER_ID}/apply_patches.sh)..."
  (cd "${RUNNER_DIR}" && bash apply_patches.sh)
fi

echo ""
echo "Done! Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
if [[ "${USE_MIRROR}" -eq 1 ]]; then
  echo "  export HF_ENDPOINT=https://hf-mirror.com"
fi
