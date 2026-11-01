# Agent Test 8: DeepSeek HIC project

This is the source bundle for the local HIC workflow and the one-pass DeepSeek Test 1 evaluation. It includes the Test 8 code, frozen manifests, run and audit artifacts, datasets, and portable Qdrant libraries. The repository `BILL_DeepSeek` branch also contains this project expanded under `HIC/` for direct browsing, plus a ZIP and per-file SHA256 manifest.

## DeepSeek Test 1 runner

The strict one-pass runner is `scripts/run_deepseek_test1_strict.py`. It binds to the `DeepSeek Test 1` collection and writes outputs to the local evaluation artifact root recorded in the runner and manifests. The collection's frozen 703-point evaluation baseline is retained at `LLM/evaluations/Agent Test 8/deepseek_test_1_20261101/preflight/qdrant_snapshot/agent_test_8_filtered_703.jsonl`.

For a portable copy of the current live retrieval collection, see `qdrant/DeepSeek_Test_1_718_20261101.snapshot`. It contains 718 points, 1024-dimensional Cosine vectors, including 15 validation-derived experiences added after scoring. This library is useful for continued HIC development; results from it are not an untouched blind benchmark. Its source, point count, Qdrant version, file size, and hashes are recorded in `qdrant/snapshot_manifest.json`.

## Setup

Use Python 3.12 and install `config/requirements.txt`. Start from `config/api.env.example`, create your local `config/api.env`, and enter your own `DEEPSEEK_API_KEY`. The package excludes `config/api.env`, model weights, virtual environments, caches, and Python bytecode. Keep API credentials out of Git.

The runner uses local `BAAI/bge-m3` for 1024-dimensional embeddings and a local Qdrant REST service at the configured endpoint. Use a compatible Qdrant version for snapshot restore (the included snapshot was made with Qdrant 1.19.2; Qdrant documents restore compatibility for the same minor version or the next minor version).

## Restore the current Qdrant library

Install the snapshot into a fresh collection name so an existing collection is not overwritten. From the project root, for example:

```powershell
curl.exe -X POST "http://127.0.0.1:6333/collections/deepseek_test_1_imported_20261101/snapshots/upload?priority=snapshot" -F "snapshot=@qdrant/DeepSeek_Test_1_718_20261101.snapshot"
```

Then verify that the restored collection is green with 718 points and a 1024-dimensional Cosine vector. If the destination already exists, choose another fresh name or inspect it before recovery. Qdrant's uploaded-snapshot endpoint can replace data in an existing collection.

## Evaluation boundaries

The 15-case result is a one-pass frozen evaluation using Test 7's development memory bank. It is not an independent external benchmark. The 718-point current library contains 15 experiences derived from those validation cases. The target meaning and units for Health, Security, and new_metr remain unknown; scores use raw numeric targets. Outer-test truth was not read. Do not present the post-score library as blind evidence.

The project retains historical runners and artifacts for audit. Read the relevant run manifest before starting a new evaluation; do not reuse a run ID or overwrite an existing frozen artifact directory.

## Windows paths

Some frozen audit filenames are long. When extracting the ZIP on Windows, use a short destination such as C:\HIC to avoid legacy path-length limits.

