# AccelMark Benchmark Skill for OpenClaw

Benchmark your AI accelerator and see how you rank in the AccelMark community.

## What it does

1. Detects your GPU/NPU automatically
2. Selects the right test for your hardware (no configuration needed)
3. Runs a 2-5 minute benchmark
4. Shows your speed and community ranking
5. Optionally submits to the AccelMark leaderboard

## Install

In OpenClaw:
```
install skill AccelMark
```

Or from ClawHub: [clawhub.ai/skills/accelmark](https://clawhub.ai/skills/accelmark)

## Usage

Just say:
- "benchmark my GPU"
- "run accelmark"
- "how fast is my machine for AI?"

## Requirements

- Python 3.10+
- 4GB+ VRAM (CPU fallback available but slow)
- Internet connection (for model download on first run)

First run downloads the model (~4GB for 8B Q4). Subsequent runs use cache.
