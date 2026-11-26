# AccelMark Documentation

**No single fastest chip — only the best serving recipe for your workload.**

AccelMark is an open, reproducible LLM inference benchmark covering **8 Roofline workload suites (A–H)** across **NVIDIA, AMD, Huawei Ascend, Apple Silicon, Google TPU, Moore Threads, and Iluvatar CoreX**.

<div class="am-cards">
  <a class="am-card" href="getting-started/first-submission/">
    <strong>Run &amp; Submit (~10 min) →</strong>
    <span>Install dependencies, pass the automatic MMLU accuracy gate, and publish your first hardware baseline.</span>
  </a>
  <a class="am-card" href="key-findings/">
    <strong>Key Findings &amp; Procurement →</strong>
    <span>Explore the 6 empirical regime reversals and the workload-to-recipe decision matrix.</span>
  </a>
  <a class="am-card" href="guides/add-a-runner/">
    <strong>Extend AccelMark →</strong>
    <span>Add a new inference engine runner, support a new accelerator family, or query our static JSON API.</span>
  </a>
</div>

---

## Quick Navigation

| I want to… | Start here |
|---|---|
| Browse live rankings & compare chips | [**Live Leaderboard ↗**](https://freedomintelligence.github.io/AccelMark/) |
| Install for NVIDIA, AMD, Ascend, Iluvatar, MTT, TPU, or Mac | [Installation](getting-started/install.md) |
| Submit without using `git` | [Submit Without Git (`--zip`)](guides/submit-without-git.md) |
| Reproduce a community run within ±5% | [Reproduce & Verify](guides/reproduce-a-result.md) |
| Understand Closed vs Open tiers & runner hashing | [Methodology & Integrity](methodology.md) |
| Query the leaderboard locally without a GPU | [Zero-GPU Advisor & JSON API](guides/api-and-advisor.md) |
| 阅读中文入门与国内镜像配置 | [中文快速上手](zh/getting-started.md) |
