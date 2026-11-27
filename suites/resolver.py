"""
Suite & Model Track resolver for AccelMark.

AccelMark keeps a fixed set of 8 Roofline workload suites (suite_A..suite_H).
Newer model generations are registered inside each suite's ``suite.json`` under
``model_tracks`` rather than proliferating new suite folders.

Both addressing forms resolve to the exact same canonical ``suite_id``:
  - ``suite_A``                             -> base suite (default model)
  - ``suite_A_qwen38_27b``                  -> base ``suite_A``, track ``qwen38_27b``
  - ``--suite suite_A --model-track qwen38_27b`` -> canonical ``suite_A_qwen38_27b``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITES_DIR = REPO_ROOT / "suites"

_SUITE_RE = re.compile(r"^(suite_[A-Z])(?:_([a-z0-9][a-z0-9_]*))?$")
_TRACK_META_KEYS = {"label", "status", "notes"}


def parse_suite_id(suite_id: str | None) -> tuple[str, str | None]:
    """Split a canonical suite_id into ``(base_suite, model_track)``.

    Examples:
      >>> parse_suite_id("suite_A")
      ('suite_A', None)
      >>> parse_suite_id("suite_A_qwen38_27b")
      ('suite_A', 'qwen38_27b')
      >>> parse_suite_id("custom")
      ('custom', None)
    """
    raw = (suite_id or "").strip()
    if not raw:
        return "", None
    m = _SUITE_RE.match(raw)
    if not m:
        return raw, None
    return m.group(1), m.group(2)


def compose_suite_id(suite_id: str, model_track: str | None = None) -> str:
    """Compose canonical suite_id from a base suite and optional model track."""
    base, existing_track = parse_suite_id(suite_id)
    track = (model_track or "").strip()
    if not track or track.lower() in ("default", "base", "legacy"):
        return suite_id
    if existing_track:
        # Already includes a track suffix; honor explicit override cleanly.
        return f"{base}_{track}"
    return f"{base}_{track}"


def short_model_label(model_id: str | None) -> str:
    """Turn a HuggingFace repo ID into a concise display label."""
    if not model_id:
        return ""
    name = str(model_id).split("/")[-1]
    name = re.sub(r"^Meta-", "", name)
    name = re.sub(r"-Instruct(?:-v\d+(?:\.\d+)*)?$", "", name)
    return name


def load_suite(
    suite_id: str,
    model_track: str | None = None,
    suites_dir: Path | None = None,
) -> dict[str, Any]:
    """Load and resolve a suite contract, merging any model track overrides.

    Raises:
      FileNotFoundError if the base suite.json does not exist.
      KeyError if ``model_track`` is not registered under the base suite.
    """
    root = Path(suites_dir) if suites_dir is not None else SUITES_DIR
    canonical_id = compose_suite_id(suite_id, model_track)

    # 1. Exact folder match (supports standard suites and any legacy standalone folder)
    direct_path = root / canonical_id / "suite.json"
    if direct_path.exists():
        data = json.loads(direct_path.read_text(encoding="utf-8"))
        base, track = parse_suite_id(canonical_id)
        resolved = dict(data)
        resolved["suite_id"] = canonical_id
        resolved.setdefault("base_suite", base)
        resolved.setdefault("model_track", track)
        resolved.setdefault(
            "model_track_label",
            short_model_label(resolved.get("model_id") or resolved.get("base_model_id")),
        )
        return resolved

    # 2. Resolve model track from base suite's suite.json
    base, track = parse_suite_id(canonical_id)
    base_path = root / base / "suite.json"
    if not base_path.exists():
        raise FileNotFoundError(f"Suite '{canonical_id}' not found at {base_path}")

    base_data = json.loads(base_path.read_text(encoding="utf-8"))
    tracks = base_data.get("model_tracks") or {}
    if not track or track not in tracks:
        avail = sorted(tracks.keys())
        avail_str = ", ".join(avail) if avail else "(none)"
        raise KeyError(
            f"Unknown model track '{track}' for {base}. "
            f"Available tracks in suites/{base}/suite.json: {avail_str}"
        )

    track_cfg = tracks[track]
    resolved = {k: v for k, v in base_data.items() if k != "model_tracks"}
    for k, v in track_cfg.items():
        if k in _TRACK_META_KEYS:
            continue
        resolved[k] = v

    resolved["suite_id"] = canonical_id
    resolved["base_suite"] = base
    resolved["model_track"] = track
    resolved["model_track_label"] = track_cfg.get("label") or short_model_label(
        resolved.get("model_id") or resolved.get("base_model_id")
    )
    resolved["model_track_status"] = track_cfg.get("status", "current")
    return resolved


def list_model_tracks(base_suite: str, suites_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return ordered list of all tracks (base default + configured model_tracks)."""
    root = Path(suites_dir) if suites_dir is not None else SUITES_DIR
    base, _ = parse_suite_id(base_suite)
    path = root / base / "suite.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks_cfg = data.get("model_tracks") or {}
    has_newer = any(
        isinstance(t, dict) and t.get("status", "current") == "current"
        for t in tracks_cfg.values()
    )
    base_model = data.get("model_id") or data.get("base_model_id") or ""
    out = [
        {
            "track_id": "default",
            "suite_id": base,
            "label": short_model_label(base_model) or base,
            "model_id": base_model,
            "status": "legacy" if has_newer else "current",
        }
    ]
    for tid, tcfg in tracks_cfg.items():
        if not isinstance(tcfg, dict):
            continue
        mid = tcfg.get("model_id") or tcfg.get("base_model_id") or base_model
        out.append(
            {
                "track_id": tid,
                "suite_id": f"{base}_{tid}",
                "label": tcfg.get("label") or short_model_label(mid) or tid,
                "model_id": mid,
                "status": tcfg.get("status", "current"),
                "notes": tcfg.get("notes"),
            }
        )
    return out
