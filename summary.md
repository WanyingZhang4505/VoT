# Research GNN — project status

Last updated: 2026-09-27
Working branch: `bill`

## DeepSeek Flash / VoT HIC status

- The local Alice/VoT repository currently contains one training example (`#611 OT`, 8 history values and 12 target values) and one test example (`#719 OT`, 8 history values). It does not contain the broader global weather, politics, or multi-country supply-chain dataset discussed elsewhere.
- `config/api.env` is local-only and must not be committed or printed. `DEEPSEEK_API_KEY` is configured; the key itself is intentionally not recorded here.
- `LLM/deepseek_client.py` sends DeepSeek chat-completion requests in JSON mode and handles transient HTTP failures and empty model content with bounded retries.
- JSON-mode requests explicitly disable DeepSeek thinking. A failed full run was diagnosed as `finish_reason=length` with reasoning content present: thinking consumed the 2,048-token output budget before final JSON was emitted. The disabled-thinking request was tested and the complete run then succeeded.
- DeepSeek Flash is used for text generation (summary, forecast/reasoning, and correction); Gemini configuration/calls remain disabled. HIC historical-case retrieval now runs locally with `BAAI/bge-small-en-v1.5` and cosine similarity, with no API request for embedding or reranking.
- Knowledge-base entries store the case summary, correction, normalized 384-dimensional BGE embedding, and embedding model ID. Legacy entries without BGE embeddings are encoded locally when retrieved; vectors tagged with another/no model ID are never mixed into BGE cosine comparisons.
- The DeepSeek knowledge base is stored separately as `LLM/knowledge_base/kb_socialgoods_deepseek.json` to preserve any legacy data; the output remains `LLM/vot_hic_output.json`.
- `LLM/template/socialgoods.json` now contains a valid generated template. A prior one-off cache write accidentally appended literal `\\n`; that file was repaired and validated with `json.loads`.
- A live DeepSeek Flash request to `summarize()` using the repository's `#611 OT` history/news sample completed successfully and returned valid JSON with the expected `dataset_info`, `predictive_factors`, and `forecast_outlook` sections.
- A full `LLM/hic.py` run completed before local BGE integration with the existing `#611 OT` training sample and `#719 OT` test sample. That prior run saved a one-entry DeepSeek-only knowledge base and a 12-step test prediction; its sole candidate was selected by the then-current DeepSeek reranker.
- The full run is a workflow test, not an accuracy result. The test sample has no ground-truth target values, and the generated summary/prediction included race-specific claims not supported by the supplied `OT` series. Improve prompt constraints and factual grounding before treating generated text as reliable annotations.

## Verification

- DeepSeek client and retrieval unit tests: 9 passed.
- The template JSON parses successfully.
- `LLM/knowledge_base/kb_socialgoods_deepseek.json` and `LLM/vot_hic_output.json` were created and parsed after the live end-to-end run.
- `git diff --check` passes; only expected Windows line-ending warnings were reported.
- No API key or other credential is stored in this file.

## Next step

Add more historical training cases and test cases with ground truth, then validate generated claims against supplied fields before reporting retrieval quality or forecasting accuracy. The current one-case retrieval is only a workflow demonstration.

## 2026-09-27 Local BGE embedding deployment

- Replaced HIC's DeepSeek candidate reranking with lazy local `BAAI/bge-small-en-v1.5` inference in `LLM/hic.py`. It uses the existing pinned `transformers==4.52.4` dependency, CLS pooling, L2 normalization, the BGE retrieval instruction on queries only, and cosine similarity; DeepSeek generation behavior remains unchanged.
- Summary text serialization is deterministic and prioritizes temporal dynamics and external drivers, followed by forecast outlook and static dataset metadata. BGE's 512-token cap is enforced with an explicit truncation warning.
- Knowledge-base entries now persist `embedding` and `embedding_model`. Legacy cases with missing/incompatible model tags are embedded on demand before comparison. Embeddings are derived from summaries only; correction notes are returned with the selected case, not included in the similarity input.
- Created an isolated, Git-ignored `Research GNN/.venv` using Python 3.12.14, CPU-only `torch==2.6.0+cpu`, `transformers==4.52.4`, PyYAML, and python-dotenv. This is the HIC-specific runtime, not the full set of repository experiment dependencies.
- Downloaded BGE weights to the standard Hugging Face cache at `C:\Users\admin\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5`; weights are outside the repository. Windows reported that symlink caching is unavailable, so Hugging Face used regular cached files.
- `scripts/test_local_bge.py` smoke-tested a real stored historical summary and the current `#719 OT` test context on CPU: 384 dimensions; historical summary 437 tokens; query with instruction 105 tokens; L2 norms 0.99999993 and 1.00000001; cosine 0.857549. These are inference/shape checks only, not retrieval-quality evidence.
- Added unit tests for deterministic serialization, nearest cosine selection, legacy KB backfill, malformed vector dimensions, persistence, and empty-KB handling. Final verification: 12 unit tests passed.
- The knowledge base still contains just one historical case; the current smoke score cannot show which case is more relevant or prove retrieval accuracy. Add diverse past cases before evaluating retrieval quality.

## 2026-09-27 End-to-end HIC run with DeepSeek Flash and local BGE

- Ran `\.venv\Scripts\python.exe LLM\hic.py` using the repository's `#611 OT` training sample and `#719 OT` test sample. The cached `socialgoods` template was loaded; template generation was not called. The first attempt was blocked by sandbox networking (`WinError 10013`); after network permission was granted, the run completed with exit code 0. The existing KB and output JSON were backed up in the system temp directory before rerunning.
- Training path completed: DeepSeek generated a structured summary, produced an initial 12-step forecast, and generated a correction from the supplied 12-value training truth. Local BGE encoded the training summary and persisted a normalized 384-D vector with its model ID in `LLM/knowledge_base/kb_socialgoods_deepseek.json`.
- Test path completed: DeepSeek summarized the `#719 OT` history/news; local BGE encoded that summary and selected the sole KB case by cosine similarity (`0.9065579`); DeepSeek returned 12 forecast values. Results are in `LLM/vot_hic_output.json`; the test case has no ground-truth target, so no accuracy can be computed.
- The client does not record token usage, so exact API cost and retry-adjusted HTTP request count were not measured. The code path makes five logical DeepSeek calls for this one-train/one-test run (train summary, initial forecast, correction, test summary, final forecast); BGE inference is local and offline.
- Important quality issue: the generated training summary asserted race-specific outcomes, named demographic groups, sector layoffs, and policy/Fed actions not present in the supplied sample. The test reasoning also refers to race disparities despite the test input providing only one OT series. This likely reflects the generic dataset description/template steering the model beyond evidence. Treat the run as an integration proof, not a trustworthy forecast or validated event analysis; next improve grounding and constrain unsupported claims.
- Alice's immediate raw-news-to-template-structured-summary step is represented, and HIC's historical correction retrieval now uses local BGE rather than Gemini embeddings. This repository version still does not implement the proposed hierarchical time-series graph/GNN extension; the current VoT samples are a single series and one training case.
