import json
import tempfile
import unittest
from pathlib import Path

from LLM.deepseek_strict_tool_provider import _atomic_json
from scripts.run_deepseek_hic_case import load_case_input, new_case_state


def valid_case():
    return {
        "case": {
            "case_id": "demo-case-001",
            "dataset": "DemoDataset",
            "frequency": "monthly",
            "history_start": "2020-01-01",
            "history_end": "2024-12-01",
            "target_start": "2025-01-01",
            "target_end": "2025-03-01",
            "target_timestamps": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "series": [10.0, 11.0, 12.0],
            "news": "Only dated information available before target_start.",
        },
        "contract": {
            "target": "unknown numeric target",
            "unit": "unknown",
            "frequency": "monthly",
            "horizon": 3,
        },
        "dataset_description": "A demonstration dataset.",
    }


class DeepSeekHicCaseInputTests(unittest.TestCase):
    def load_value(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return load_case_input(path)

    def test_accepts_truth_blind_case_and_builds_template(self):
        loaded = self.load_value(valid_case())
        self.assertEqual(loaded["case"]["series"], [10.0, 11.0, 12.0])
        self.assertEqual(loaded["contract"]["horizon"], 3)
        self.assertIn("dataset_info", loaded["template"])
        self.assertEqual(len(loaded["input_sha256"]), 64)

    def test_rejects_truth_fields_anywhere_in_input(self):
        value = valid_case()
        value["case"]["covariates"] = {"evaluation": {"observed_targets": [1, 2, 3]}}
        with self.assertRaisesRegex(ValueError, "truth field"):
            self.load_value(value)

    def test_rejects_non_finite_series_values(self):
        value = valid_case()
        value["case"]["series"] = [1.0, float("nan")]
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.load_value(value)

    def test_rejects_horizon_disagreement_with_target_dates(self):
        value = valid_case()
        value["contract"]["horizon"] = 4
        with self.assertRaisesRegex(ValueError, "exactly contract.horizon dates"):
            self.load_value(value)

    def test_initial_state_leaves_hic_identity_unset_for_first_advance(self):
        state = new_case_state("run-1", "case-1")
        self.assertNotIn("input_sha256", state)
        self.assertFalse(state["truth_read"])
        self.assertEqual(state["qdrant_writes"], 0)

    def test_atomic_provider_record_uses_short_temp_name_in_deep_run_path(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory)
            for index in range(3):
                nested /= f"case-artifact-directory-{index:02d}"
            target = nested / ("a" * 64 + ".json")
            self.assertLess(len(str(target)), 259)
            _atomic_json(target, {"status": "in_flight"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             {"status": "in_flight"})


if __name__ == "__main__":
    unittest.main()
