# DeepSeek HIC: setup, Qdrant library, and new-case predictions

This guide explains how to clone the public `BILL_DeepSeek` branch, restore its Qdrant library, configure DeepSeek and local BGE-M3, and run one new case at a time. It describes the code shipped in this branch as of 2026-11-01.

The active path for a user-supplied case is `scripts/run_deepseek_hic_case.py`. This branch is organized around that reusable one-case workflow.

## 1. What is included

- `scripts/run_deepseek_hic_case.py`: truth-blind one-case runner. It generates one case summary, a no-retrieval forecast (A), embeds the summary locally with BGE-M3, retrieves one direct-nearest Top-1 Qdrant memory, then generates a fresh Full HIC forecast (B). It does not write points to Qdrant.
- `qdrant/DeepSeek_Test_1_718_20261101.snapshot`: portable snapshot of the live `DeepSeek Test 1` collection at 718 points, 1024-dimensional Cosine vectors, created with Qdrant 1.19.2.
- `qdrant/snapshot_manifest.json`: source collection, count, vector configuration, snapshot size, creation time, and SHA256.

The 718-point snapshot includes 15 validation-derived experiences added after the 15-case evaluation. It is a development memory bank, not an untouched blind-test library. Do not use it to claim blind performance on those same 15 cases. The HIC and Qdrant code use the collection read-only during a new-case prediction. Frozen Agent Test 8 case outputs, score logs, and specialized evaluation runners are not included because they are not needed to run a new case.

## 2. Software and model downloads

Use Python 3.12 on Windows for the pinned dependency set. The local preflight for this runner was verified with Python 3.13.1, PyTorch 2.6.0+cpu, and Transformers 4.52.4; the official Windows PyTorch guidance currently lists Python 3.9–3.12, so Python 3.12 is the safer clone setup.

- [Python for Windows](https://www.python.org/downloads/windows/)
- [Git for Windows](https://git-scm.com/install/windows)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
- [PyTorch 2.6.0 CPU wheel command](https://pytorch.org/get-started/previous-versions/)
- [BAAI/bge-m3 model files and model card](https://huggingface.co/BAAI/bge-m3)
- [Qdrant installation](https://qdrant.tech/documentation/install/)
- [Qdrant snapshot restore guide](https://qdrant.tech/documentation/operations/snapshots/)
- [DeepSeek API quick start](https://api-docs.deepseek.com/quick_start/agent_integrations/)
- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek Platform](https://platform.deepseek.com/) — create your own API key in the account dashboard.

The model weights are not stored in Git. Download BGE-M3 once while Hugging Face is reachable; prediction and preflight then use the local cache in offline mode. BGE-M3 produces 1024-dimensional vectors and supports up to 8192 input tokens according to its model card.

## 3. Clone the branch and install Python dependencies

Run these commands in PowerShell. The branch is public, so GitHub sign-in is not required to clone it.

```powershell
git clone --branch BILL_DeepSeek --single-branch https://github.com/WanyingZhang4505/VoT.git
Set-Location .\VoT\HIC

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r config/requirements.txt
```

The CPU PyTorch command is the official 2.6.0 wheel command. If you want a CUDA build, use the matching 2.6.0 command from the PyTorch previous-versions page, then install the remaining requirements. Keep Python 3.12 for this pinned environment; newer Python versions may not have compatible wheels for all pinned dependencies.

Download BGE-M3 into the Hugging Face cache:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BAAI/bge-m3'))"
```

The first download needs internet access. The runner sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when it loads the encoder, so it will fail clearly if the local model files are missing.

## 4. Start Qdrant and restore the 718-point collection

The included `docker-compose.yml` pins Qdrant to 1.19.2 and binds REST to `127.0.0.1:6333` and gRPC to `127.0.0.1:6334`. REST is the interface used by this runner.

First check whether a suitable Qdrant service is already listening:

```powershell
try {
    Invoke-WebRequest -Uri http://127.0.0.1:6333/readyz -TimeoutSec 5 | Select-Object StatusCode
} catch {
    Write-Output "No ready Qdrant service on 127.0.0.1:6333"
}
```

If a healthy service already owns port 6333, reuse it and skip `docker compose up`. Otherwise start the bundled single-node service from the `HIC` directory:

```powershell
docker compose up -d
Invoke-WebRequest -Uri http://127.0.0.1:6333/readyz -TimeoutSec 30 | Select-Object StatusCode
```

Restore the snapshot into a fresh collection named `deepseek_test_1_imported_20261101`. The upload endpoint can replace data if the target collection already exists, so the commands below stop if that name is present.

```powershell
$collectionName = "deepseek_test_1_imported_20261101"
$collectionList = Invoke-RestMethod -Uri http://127.0.0.1:6333/collections -TimeoutSec 10
$existingNames = @($collectionList.result.collections | ForEach-Object { $_.name })
if ($existingNames -contains $collectionName) {
    throw "Collection $collectionName already exists. Inspect it or choose a new name before restore."
}

Get-FileHash .\qdrant\DeepSeek_Test_1_718_20261101.snapshot -Algorithm SHA256
curl.exe -X POST "http://127.0.0.1:6333/collections/$collectionName/snapshots/upload?priority=snapshot" `
  -F "snapshot=@qdrant/DeepSeek_Test_1_718_20261101.snapshot"
```

Expected snapshot SHA256:

```text
d0fd9a858bc75f76504da492b9d72e152c867e8c69a541736842ae3c42c32860
```

Verify the restored collection before prediction:

```powershell
$info = Invoke-RestMethod -Uri "http://127.0.0.1:6333/collections/$collectionName" -TimeoutSec 10
$info.result | Select-Object status, points_count
$info.result.config.params.vectors | ConvertTo-Json -Depth 4
```

Expected values are `green`, `718`, vector size `1024`, and distance `Cosine`. The snapshot was produced by Qdrant 1.19.2. For compatibility, restore with 1.19.2 or a Qdrant version supported by the official snapshot compatibility guidance.

The published snapshot was restored into a fresh verification collection on 2026-11-01. All 718 point IDs, payloads, and vectors matched the source collection. The saved audit is `HIC/qdrant/restore_verification_20261101.json` (repository-root path).

## 5. Configure the DeepSeek API key

Create the local ignored key file and edit it in a text editor:

```powershell
Copy-Item .\config\api.env.example .\config\api.env
notepad .\config\api.env
```

Set the line below to the key from your own DeepSeek account:

```dotenv
DEEPSEEK_API_KEY=your_key_here
```

Replace the placeholder locally. Do not commit or share `config/api.env`. The repository ignores this file. The runner reads it through `LLM/hic.py`; the key is not written to prediction artifacts. The shipped model configuration is `deepseek-flash`; local BGE-M3 performs embeddings and does not use a DeepSeek embedding API.

## 6. Run preflight without making a model request

Store user-created cases under `local_cases/`; that directory and all new-case run outputs are ignored by Git. Create `local_cases/new_case.json` using the schema below, then check local dependencies and the restored Qdrant collection:

```powershell
& .\.venv\Scripts\python.exe scripts\run_deepseek_hic_case.py preflight `
  --collection deepseek_test_1_imported_20261101 `
  --expected-points 718 `
  --qdrant-url http://127.0.0.1:6333 `
  --output "LLM\evaluations\DeepSeek HIC New Cases\preflight_20261101.json"
```

Preflight checks the collection status/count/vector configuration, the offline BGE-M3 model and vector dimension, and whether a DeepSeek API key is configured. It makes no DeepSeek request, reads no truth, and writes no Qdrant points. A successful result reports `ready: true`.

## 7. Prepare a new case without future truth

The input file must contain only what is available at the forecast cutoff. `series` is the observed numeric history. `news` and `covariates` may contain only evidence actually available by that cutoff. Do not include future target values, a forecast, a prior score, or keys named `truth`, `observed_targets`, `test_truth`, `prediction`, `mae`, `mse`, `rmse`, `score`, or equivalent.

`target` and `unit` describe the numeric target. If their meaning is unknown, use `unknown`; do not infer meaning or units from a dataset name or news text. `target_timestamps` must have exactly `horizon` entries and its first and last dates must match `target_start` and `target_end`.

Example `local_cases/new_case.json`:

```json
{
  "case": {
    "case_id": "my_new_case_001",
    "dataset": "MyDataset",
    "frequency": "monthly",
    "history_start": "2024-08-01",
    "history_end": "2024-12-01",
    "target_start": "2025-01-01",
    "target_end": "2025-03-01",
    "target_timestamps": ["2025-01-01", "2025-02-01", "2025-03-01"],
    "series": [10.0, 10.4, 9.8, 11.2, 12.0],
    "news": "Dated source text available before 2025-01-01.",
    "covariates": {}
  },
  "contract": {
    "target": "unknown",
    "unit": "unknown",
    "frequency": "monthly",
    "horizon": 3
  },
  "dataset_description": "Describe the data source and observable context without claiming an unknown target meaning."
}
```

Use a different, unique `case_id` for every event. Keep the input JSON outside the repository if it contains private information; `local_cases/` is ignored as a convenience.

## 8. Predict one case with A/B outputs

Run cases serially, one command per case, and use a unique `run-id`:

```powershell
& .\.venv\Scripts\python.exe scripts\run_deepseek_hic_case.py predict `
  --input local_cases\new_case.json `
  --collection deepseek_test_1_imported_20261101 `
  --expected-points 718 `
  --qdrant-url http://127.0.0.1:6333 `
  --run-id my_new_case_001_20261101
```

For one case the runner makes three main DeepSeek requests: one summary, one No Retrieval A forecast, and one independently generated Full HIC B forecast. B uses the same case summary and history as A, plus the correction text from one direct-nearest Top-1 point. Forecasts use strict function calling with exactly the requested number of numeric fields. Local BGE-M3 creates the query vector; Qdrant returns the nearest point.

The runner performs no automatic correction learning, scoring, truth read, or Qdrant write. It records the request/response evidence and forecast state under:

```text
LLM/evaluations/DeepSeek HIC New Cases/<run-id>/
```

Key files are `run_manifest.json`, `case_input.json`, `state.json`, `result.json`, and `provider_calls/*.json`. `result.json` includes A and B arrays, the retrieved point ID and source case ID, similarity, source metadata, input hash, provider protocol, and prompt version. Provider-call records preserve prompts and raw responses. These outputs may contain the case's input text, so keep the output directory private.

To process another case, provide its own input file and run ID. Each run reuses the same 718-point library because the runner is read-only. It does not add the new prediction or outcome as a correction.

## 9. Retry an interrupted run

If a technical error occurs after the run directory is created, keep the same input and run ID and resume:

```powershell
& .\.venv\Scripts\python.exe scripts\run_deepseek_hic_case.py predict `
  --input local_cases\new_case.json `
  --collection deepseek_test_1_imported_20261101 `
  --expected-points 718 `
  --qdrant-url http://127.0.0.1:6333 `
  --run-id my_new_case_001_20261101 `
  --resume
```

Resume verifies the original input hash, collection, and Qdrant URL. Completed provider calls are reused; unresolved calls may be retried with the same prompt/settings. Do not edit the input under the same run ID. A completed `result.json` is returned as-is; use a new run ID for a genuinely new case.

## 10. Understand retrieval and the result boundary

This new-case command deliberately uses the same direct-nearest Top-1 rule as the DeepSeek Test 1 path. It does not filter by date, horizon, source family, or source case. Inspect the saved Top-1 `case_id`, `point_id`, `source_metadata`, and similarity for every prediction. The nearest point can be from the same source family or have a horizon/time relationship that is not suitable for the new case.

The 718-point snapshot includes the 15 experiences derived from the previous validation results. A new case that duplicates one of those cases can retrieve its own validation-derived correction. Treat such a run as a workflow demonstration, not a blind evaluation. Use `run_deepseek_hic_case.py` for new cases; this branch contains the reusable one-case runner, not the fixed 15-case evaluation protocol.

No score is calculated for a new case because the case file contains no truth. If an outcome becomes available later, keep it sealed until the A/B outputs and request records are saved. Any later correction-writing process changes the library and must be recorded as development feedback; it is not part of this read-only prediction command.

## 11. Common checks

| Symptom | Check |
|---|---|
| `/readyz` fails | Start Docker Desktop or inspect the existing Qdrant service on port 6333. Do not start another Qdrant instance on an occupied port. |
| Restore says collection exists | Choose a fresh collection name, update `--collection`, and keep the existing collection untouched. |
| Preflight reports wrong count/config | Restore the included 718-point snapshot to a fresh name; verify green, 718, 1024, Cosine. |
| BGE-M3 unavailable in offline mode | Run the `snapshot_download` command in section 3 while internet is available, then rerun preflight. |
| API key missing | Edit local `config/api.env` and confirm `DEEPSEEK_API_KEY` is non-empty without printing the key. |
| Forecast length rejected | Check `contract.horizon` equals the length of `case.target_timestamps`; keep every timestamp strictly increasing. |
| Retrieval returns no correction | Inspect the selected point and collection contents. The runner fails rather than silently using a different memory. |

## 12. Official technical references

- [Qdrant snapshot upload and compatibility](https://qdrant.tech/documentation/operations/snapshots/)
- [Qdrant Docker installation](https://qdrant.tech/documentation/install/)
- [BGE-M3 model card and supported dimensions/context length](https://huggingface.co/BAAI/bge-m3)
- [DeepSeek API model and first-call guide](https://api-docs.deepseek.com/quick_start/agent_integrations/)
- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
