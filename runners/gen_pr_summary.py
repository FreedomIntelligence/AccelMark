#!/usr/bin/env python3
"""
Generate a markdown summary table from a result.json file.
Used by .github/workflows/process_submissions.yml to build PR descriptions.
Usage: python runners/gen_pr_summary.py <path/to/result.json>
"""
import json, sys

with open(sys.argv[1]) as f:
    r = json.load(f)

chip    = r.get('chip', {})
model   = r.get('model', {})
sw      = r.get('software', {})
metrics = r.get('metrics', {})
acc     = r.get('accuracy') or {}
meta    = r.get('meta', {})
suite   = r.get('suite_id', '')
dash    = '—'

# ── Primary throughput — varies by suite ─────────────────────────────────────
thr = None

# Suite A/B/D: standard offline block
offline = metrics.get('offline', {})
rows    = (offline.get('results_by_concurrency') or
           offline.get('results_by_batch_size') or [])
valid   = [x for x in rows if not x.get('oom') and x.get('throughput_tokens_per_sec')]
if valid:
    thr = round(max(x['throughput_tokens_per_sec'] for x in valid))

# Suite C: best throughput across precision formats
if thr is None:
    quant = metrics.get('quantization', {})
    entries = quant.get('results_by_precision', [])
    vals = [e.get('best_throughput_tokens_per_sec') for e in entries
            if e.get('best_throughput_tokens_per_sec')]
    if vals:
        thr = round(max(vals))

# Suite E: base throughput from scaling block
if thr is None:
    scaling = metrics.get('scaling', {})
    base = scaling.get('base_throughput_tokens_per_sec')
    if base:
        thr = round(base)
    else:
        for entry in scaling.get('results_by_chip_count', []):
            if entry.get('chip_count') == 1:
                t = entry.get('best_throughput_tokens_per_sec')
                if t:
                    thr = round(t)
                break

# ── Online / interactive ──────────────────────────────────────────────────────
online = metrics.get('online', {})
iv     = metrics.get('interactive', {})
qps    = online.get('max_valid_qps')
ttft   = iv.get('ttft_ms_p99')
# Treat 0 or None as no result
if not qps:
    qps = None

# ── Suite C: accuracy per format ─────────────────────────────────────────────
if suite == 'suite_C':
    quant   = metrics.get('quantization', {})
    entries = quant.get('results_by_precision', [])
    bf16    = next((e for e in entries if e.get('precision') == 'BF16'), {})
    acc_val = f"{bf16.get('accuracy_score', dash)} (BF16)" if bf16 else dash
elif suite == 'suite_E':
    acc_val = dash   # Suite E does not run accuracy
elif acc:
    acc_val = acc.get('valid', dash)
else:
    acc_val = dash

# ── Suite E: scaling efficiency ───────────────────────────────────────────────
scaling_note = ''
if suite == 'suite_E':
    scaling = metrics.get('scaling', {})
    parts   = []
    for entry in scaling.get('results_by_chip_count', []):
        count = entry.get('chip_count')
        eff   = entry.get('scaling_efficiency')
        if count and count > 1 and eff is not None:
            parts.append(f'{count}×: {round(eff * 100)}%')
    scaling_note = ', '.join(parts) if parts else dash

# ── Print table ───────────────────────────────────────────────────────────────
print('| Field | Value |')
print('|---|---|')
print(f"| Chip | {chip.get('name', dash)} |")
print(f"| Chip count | {chip.get('count', 1)} |")
print(f"| Suite | {suite or dash} |")
print(f"| Runner | {r.get('implementation_id', dash)} |")
print(f"| Framework | {sw.get('framework', dash)} {sw.get('framework_version', '')} |")
print(f"| Model | {model.get('model_id', dash)} |")
print(f"| Precision | {model.get('precision', dash)} |")
print(f"| Offline throughput | {thr if thr else dash} tok/s |")
if suite == 'suite_E' and scaling_note:
    print(f"| Scaling efficiency | {scaling_note} |")
else:
    print(f"| Online max QPS | {round(qps, 1) if qps is not None else dash} |")
    print(f"| Interactive TTFT p99 | {round(ttft, 1) if ttft is not None else dash} ms |")
print(f"| Accuracy valid | {acc_val} |")
print(f"| Submitted by | {meta.get('submitted_by', dash)} |")
print(f"| Date | {meta.get('date', dash)} |")