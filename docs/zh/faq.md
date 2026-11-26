# 中文常见问题（FAQ）

### 需要调出 SOTA 极限速度才能提交吗？

**完全不需要。** AccelMark 排行榜默认展示的是 **Closed（Reference 参考栈）**，衡量开箱即用、可复现的真实基线。无论是消费级 4090/3090、老卡 V100/T4，还是国产加速卡，诚实的默认跑分都非常有价值。

### 为什么我很快的显卡在 Suite A Online 上显示 `SLA fail`（未达 SLA）？

`Online` 场景要求在泊松到达并发下满足 **`p99 TTFT ≤ 500 ms`**。24–32 GB 显存的显卡（如 RTX 5090 / 4090）虽然离线吞吐极高，但 KV-Cache 容量与调度队列压力会导致 p99 首字延迟超过 500ms 门槛——这正是 AccelMark 揭示的核心实证发现之一（离线最快 ≠ 生产 SLA 最强），并不是报错。

### `Closed (Reference)` vs `Open` 和 `community` vs `verified` 有什么区别？

- **Runner 层级（Closed / Open）** 区分代码栈：`reference` 是标准开箱栈（榜单默认显示）；`optimized` / `experimental` 是深度魔改或实验内核，可一键切换查看，不会污染标准排名。
- **Submission 信任等级（`community` → `verified`）** 区分复现度：PR 通过自动 CI 校验即刻以 `community` 上榜；由另一位贡献者在同硬件、同 Runner 下独立复现且误差 **≤ ±5%** 后晋升为 **`verified`**（榜单显示绿色盾牌）。

### 不熟悉 Git / 不想提 PR 怎么提交结果？

在 `--submit` 命令后加上 `--zip`：

```bash
python run.py --submit results/community/<run_dir> --zip
```

然后将生成的 `.zip` 压缩包直接拖拽上传到 [**GitHub 免 Git 提交表单**](https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.yml) 即可。

### 为什么新提交的 Runner PR 会被 CI 拒绝？

常见原因有 3 个：① 文件夹末尾 8 位哈希与 `runner.py` 的 SHA-256 不一致（运行 `python runners/hash_runner.py` 核对）；② 仓库内夹带了完整的上游推理框架源码（请改用 `requirements.txt` 安装）；③ 缺少 `meta.json` 字段。本地可先运行 `python runners/validate_runners.py` 自检。
