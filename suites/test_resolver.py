"""Unit tests for suites/resolver.py (Model Tracks inside fixed 8 suites)."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suites.resolver import (
    compose_suite_id,
    list_model_tracks,
    load_suite,
    parse_suite_id,
)


class TestSuiteResolver(unittest.TestCase):
    def test_parse_and_compose(self):
        self.assertEqual(parse_suite_id("suite_A"), ("suite_A", None))
        self.assertEqual(parse_suite_id("suite_A_qwen38_27b"), ("suite_A", "qwen38_27b"))
        self.assertEqual(parse_suite_id("suite_F_qwen3_06b"), ("suite_F", "qwen3_06b"))
        self.assertEqual(compose_suite_id("suite_A", None), "suite_A")
        self.assertEqual(compose_suite_id("suite_A", "default"), "suite_A")
        self.assertEqual(compose_suite_id("suite_A", "qwen38_27b"), "suite_A_qwen38_27b")
        self.assertEqual(compose_suite_id("suite_F", "qwen3_06b"), "suite_F_qwen3_06b")

    def test_load_base_and_track_suites(self):
        base_a = load_suite("suite_A")
        self.assertEqual(base_a["suite_id"], "suite_A")
        self.assertEqual(base_a["base_suite"], "suite_A")
        self.assertEqual(base_a["model_id"], "meta-llama/Meta-Llama-3-8B-Instruct")

        qwen_a = load_suite("suite_A", "qwen38_27b")
        self.assertEqual(qwen_a["suite_id"], "suite_A_qwen38_27b")
        self.assertEqual(qwen_a["base_suite"], "suite_A")
        self.assertEqual(qwen_a["model_track"], "qwen38_27b")
        self.assertEqual(qwen_a["model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(qwen_a["dataset"], base_a["dataset"])
        self.assertEqual(qwen_a["online_sla_ttft_ms"], base_a["online_sla_ttft_ms"])
        self.assertNotIn("model_tracks", qwen_a)

        # Direct canonical suite_id resolves identically
        qwen_a_direct = load_suite("suite_A_qwen38_27b")
        self.assertEqual(qwen_a, qwen_a_direct)

        qwen_f = load_suite("suite_F_qwen3_06b")
        self.assertEqual(qwen_f["suite_id"], "suite_F_qwen3_06b")
        self.assertEqual(qwen_f["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(qwen_f["dataset"], "sharegpt_edge_v1")

    def test_unknown_track_raises_key_error(self):
        with self.assertRaises(KeyError):
            load_suite("suite_A", "nonexistent_track")

    def test_list_model_tracks(self):
        tracks_f = list_model_tracks("suite_F")
        ids = [t["suite_id"] for t in tracks_f]
        self.assertIn("suite_F", ids)
        self.assertIn("suite_F_qwen3_06b", ids)


if __name__ == "__main__":
    unittest.main()
