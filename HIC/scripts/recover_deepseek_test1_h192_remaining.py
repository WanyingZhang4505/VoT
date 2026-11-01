"""Recover remaining malformed H192 cases in a separate audit lane."""

from __future__ import annotations

import recover_deepseek_test1_h192_shape as recovery


recovery.CASE_ORDINALS = (13, 14)
recovery.REPAIR_PROTOCOL = "deepseek_test1_h192_shape_recovery_cases13_14_v1"
recovery.REPAIR_DIR = recovery.HIC_ROOT / (
    "LLM/evaluations/Agent Test 8/"
    "deepseek_test_1_20261101_r2/h192_shape_repair_cases13_14"
)


if __name__ == "__main__":
    recovery.main()
