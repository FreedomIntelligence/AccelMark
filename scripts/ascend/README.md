# Huawei Ascend Platform Scripts

Benchmark scripts for Huawei Ascend NPUs using MindIE.

## Requirements

```bash
pip install -r scripts/ascend/requirements.txt
```

CANN (Compute Architecture for Neural Networks) toolkit required.
See Huawei's official documentation for CANN installation.

## MindIE (run_mindie.py)

Supports Suite A (offline, online, interactive).

```bash
python scripts/ascend/run_mindie.py \
    --suite suite_A \
    --scenario offline \
    --output-dir ./my_submission/ \
    --npu-ids 0
```

## Notes

- MindIE is Huawei's inference engine for Ascend NPUs.
- The script uses `npu-smi` for device queries — ensure it is in your PATH.
- Set `ASCEND_VISIBLE_DEVICES` or use `--npu-ids` to select devices.
