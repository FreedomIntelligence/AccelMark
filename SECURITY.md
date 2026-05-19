# Security Policy

## Scope

AccelMark is a benchmarking framework. The "interesting" security surface
is correspondingly small, but two areas matter:

1. **Code that runs locally on contributor / maintainer machines.**
   The repository ships Python that reads model files, parses third-party
   tool output (`nvidia-smi`, `rocm-smi`, `npu-smi`, `mthreads-gmi`, etc.),
   reads YAML configuration, and runs inference frameworks (vLLM, SGLang,
   mlx-lm, …) under their own dependency stacks. A malicious config,
   meta.json, or runner.py landing in `main` could compromise anyone who
   pulls and runs the repo.

2. **Submitted results.**
   `results/community/**` is community-contributed JSON. A malicious
   `result.json` cannot execute code on its own, but it can poison the
   leaderboard if the validator can be bypassed. Bugs in
   `runners/validate_submission.py` that allow obviously-fake results to
   merge are treated as security issues.

Outside of those two surfaces (in particular: bugs that produce wrong
benchmark *numbers* without a reproducibility problem) are normal bugs and
should be reported via a regular GitHub issue.

## Supported versions

AccelMark is pre-1.0 and ships from `main`. The latest commit on `main` is
the only "supported" version; we backport fixes to release tags only after
1.0.

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security report.**

Use GitHub's [private security advisory][advisory] form on this repository.
A maintainer will respond within **7 days** acknowledging the report and
providing an initial assessment. We aim to publish a fix and credit the
reporter within **30 days** of acknowledgement; if a fix is going to take
longer we will say so in the response.

[advisory]: https://github.com/FreedomIntelligence/AccelMark/security/advisories/new

When reporting, please include:

* The version (commit SHA on `main`, or release tag).
* A minimal reproduction — config files, the exact command, and the
  observed behaviour. For supply-chain reports, the offending dependency
  and version.
* Your assessment of the impact (e.g. "arbitrary file read at runner
  startup", "validator accepts result with mismatched chip name", …).

We do not currently run a paid bug bounty, but we are happy to credit
reporters in the release notes for the fix.

## What is *not* a vulnerability

For clarity, the following are explicitly out of scope:

* **Results you disagree with.** Use the *Challenge a Result* GitHub
  issue template; this is a leaderboard-policy matter, not a security one.
* **A runner that performs poorly on your hardware.** Open a regular issue
  or PR.
* **Resource exhaustion when running a benchmark you started yourself.**
  Benchmarks intentionally saturate the device; OOM and similar are
  expected operating conditions.
* **Dependencies of a runner being slow / outdated.** The runner author
  pins versions in `requirements.txt`; submit a PR for a new runner with
  updated pins (immutability rule — see `runners/README.md`).
