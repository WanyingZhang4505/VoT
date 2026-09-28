# VoT

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.10), install the dependencies with the following command:

```
conda create -n vot python==3.10
pip install -r requirements.txt
```
And you need to download the required language model in the `language_model` directory and set the corresponding `--llm_dim` parameter in `run_vot.py`

### HIC workflow: DeepSeek generation + local BGE retrieval

Set `DEEPSEEK_API_KEY=...` in the local `config/api.env` file (ignored by Git).
DeepSeek Flash generates case summaries, forecasts, and correction notes.
Historical-case retrieval is local: `BAAI/bge-small-en-v1.5` encodes each case
summary into a normalized 384-dimensional vector, and HIC selects the closest
stored summary by cosine similarity. Retrieval does not call DeepSeek or require
an embedding API key.

The query summary receives BGE's recommended retrieval instruction; stored case
summaries do not. Summary serialization puts temporal and external-event signals
before static dataset metadata. Inputs are capped at 512 tokens; longer inputs
are truncated with a warning. Embeddings are stored with each knowledge-base
entry and tagged with the model ID so vectors from another model are not mixed.
Legacy entries without BGE vectors are embedded when retrieval needs them.

On Windows, the local HIC environment used for this integration can be recreated
with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install transformers==4.52.4 PyYAML python-dotenv
```

The first BGE run downloads the model weights (about 133 MB) into the user's
Hugging Face cache, outside this repository. To smoke-test only the local
embedding and retrieval path without making DeepSeek requests, run:

```powershell
.\.venv\Scripts\python.exe scripts/test_local_bge.py
```

Run the available one-train/one-test example with:

```shell
python LLM/hic.py
```

With the template cached, the current one-train/one-test run makes five main
DeepSeek Flash chat requests (training summary, initial forecast, correction,
test summary, and final forecast); automatic retries can increase the count and
API usage charges. Local BGE handles retrieval separately. Output is written to
`LLM/vot_hic_output.json`, and the knowledge base (including vectors) to
`LLM/knowledge_base/kb_socialgoods_deepseek.json`. The repository currently has
only one training case and one test case; this verifies the workflow, not
retrieval quality or forecast accuracy. A single candidate is selected by
definition, so meaningful retrieval evaluation requires more historical cases.
The demo treats `train_samples` in `LLM/hic.py` as the complete authoritative
training corpus and rebuilds the knowledge base from it on each run. Add the full
historical corpus there before running; previously generated KB entries are not
incrementally retained, which also prevents duplicate entries on repeated runs.

### Train and evaluate model

We provide the experiment scripts for all benchmarks under the folder `./scripts/vot.sh`. You can reproduce an experiment result as the following:

```shell
bash scripts/vot.sh
```
