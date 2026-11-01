# DeepSeek HIC runtime

This directory contains the source code and the 718-point Qdrant snapshot used for read-only historical correction on new cases. Start with the [setup and new-case guide](../DEEPSEEK_HIC_README.md) in the repository root.

The active runner is `scripts/run_deepseek_hic_case.py`. The detailed guide covers dependencies, model download, Qdrant restore, API-key setup, truth-free input format, preflight, prediction, and resume.

The snapshot is in `qdrant/DeepSeek_Test_1_718_20261101.snapshot`; collection details and SHA256 are in `qdrant/snapshot_manifest.json`. It contains validation-derived development memories, so it is not an untouched benchmark library.
