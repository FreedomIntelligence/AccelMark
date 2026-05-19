#!/usr/bin/env bash
# Install dependencies from requirements.txt in three stages.
# pip cannot resolve vllm and mistral-common[image] in a single install pass.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${RUNNER_DIR}/requirements.txt"
EXTRA=()
if [[ -n "${PYTORCH_INDEX:-}" ]]; then
  EXTRA=(--extra-index-url "${PYTORCH_INDEX}")
fi

line() { awk -v p="$1" '$0 ~ "^" p "[=<>]" { print; exit }' "${REQ}"; }

echo "==> $(line mistral-common)"
pip install "$(line mistral-common)"

echo "==> $(line vllm)"
pip install "$(line vllm)" "${EXTRA[@]}"

TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT
awk '!/^#/ && NF && $0 !~ /^mistral-common/ && $0 !~ /^vllm/' "${REQ}" > "${TMP}"
echo "==> AccelMark utilities"
pip install -r "${TMP}"

python -c "import vllm; print('OK — vllm', vllm.__version__)"
