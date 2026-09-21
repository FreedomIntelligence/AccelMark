#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$runner_dir/patches/vllm_musa_skip_empty_broadcast.patch"
site_packages="$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
target="$site_packages/vllm/distributed/communication_op.py"

if grep -q 'if tensor.numel() > 0:' "$target"; then
    echo "vLLM-MUSA empty-tensor broadcast patch already applied: $target"
    exit 0
fi

patch --batch --forward -p1 -d "$site_packages" < "$patch_file"
python -m py_compile "$target"
echo "Applied vLLM-MUSA empty-tensor broadcast patch: $target"
