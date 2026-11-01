# VoT paper, repository and HIC configuration coverage

Audit date: 2026-10-30. This is a source/configuration inventory, not a new forecasting run.

## Confirmed scope

- Paper: [arXiv 2603.15452](https://arxiv.org/pdf/2603.15452), sections 3.3, 3.4, 4.1 and Appendix A.1–A.4. The paper links [decisionintelligence/VoT](https://github.com/decisionintelligence/VoT).
- Alice's [LLM/hic.py](https://github.com/WanyingZhang4505/VoT/blob/main/LLM/hic.py) implements template → summary → prediction → truth-informed historical correction → summary embedding → nearest correction retrieval. It contains one hard-coded historical sample and one query sample, rather than an exhaustive benchmark case manifest. It calls Gemini; this is not the current Codex/BGE-M3/Qdrant runner.
- The paper evaluates 10 datasets with four horizons each: **40 dataset × horizon configurations**. Monthly horizons are 6/8/10/12; Energy/Health use 12/24/36/48; Environment/Weather use 48/96/192/336. Social Good is included in the results and monthly-horizon list, although the Appendix A.1 dataset bullet list omits its individual bullet.
- Both repositories' current `scripts/vot.sh` enumerate these families, plus Economy_sort as the chronological Economy variant. Only SocialGood is uncommented in the dataset launch block. Enumeration is not proof that every benchmark has already run.
- The local eligible scope is **eight source families × four horizons = 32**, excluding four Environment horizons because the existing plan lacks their target contract, and four Traffic horizons because the source CSV is absent. Economy/Economy_sort are not counted as two independent datasets.
- The public repositories' top-level `data/` listing has no Traffic.csv. A code configuration or paper result does not establish public availability of a complete case manifest.

Sources: [Alice benchmark script](https://github.com/WanyingZhang4505/VoT/blob/main/scripts/vot.sh), [paper repository script](https://github.com/decisionintelligence/VoT/blob/main/scripts/vot.sh), [paper repository data](https://github.com/decisionintelligence/VoT/tree/main/data). URLs use main and may change; existing local source provenance remains pinned in `data/source_manifest.json`.

## Coverage of the local 32 configurations

| Family | Frozen Test 3 horizons | Added Test 6 horizons | Remaining horizons |
|---|---|---|---|
| Agriculture | 6, 8, 10, 12 | — | — |
| Climate | 6, 8, 10, 12 | — | — |
| Economy_sort | 6, 8, 10, 12 | — | — |
| Energy | 12, 24 | 36, 48 | — |
| SocialGood | 6, 8 | 10, 12 | — |
| Health | — | — | 12, 24, 36, 48 |
| Security | — | — | 6, 8, 10, 12 |
| new_metr / Weather | — | — | 48, 96, 192, 336 |
| Configuration count | 16 | 4 | 12 |

The frozen Test 3 source has 575 historical case records; the four added Test 6 configurations have 33 scored online development cases. Together these account for 608 source records across 20 configurations. A case is identified by configuration plus case ID; configuration counts, unique origins and repeated forecasts are different denominators.

Correction to the previous chat answer: the **538/575 complete, 37 planned** snapshot belongs to paused **Test 5**, stored under a directory named `Agent Test 3/hic_agent_test3_vnext_seeded_r1_20261025`. It is not the completion status of the frozen Test 3 575-record source or the Test 6 replay. The directory name was misleading; `summary.md` labels its active collection as agent_test_5.

## Fresh target-blind inventory of the remaining 12

Recomputed with `load_native_hic_splits(include_outer_test_targets=False, include_validation_truth=False, include_empty_news=False)`. No model requests, embedding calls, or Qdrant writes. Assertions checked deferred outer-test, absent validation truths and zero validation target values parsed. Saved metadata contains dates, IDs, source hashes and counts, without prediction/truth arrays.

| Dataset | Horizon | Fit windows with text | Validation windows with text |
|---|---:|---:|---:|
| Health | 12 | 51 | 11 |
| Health | 24 | 24 | 4 |
| Health | 36 | 15 | 3 |
| Health | 48 | 11 | 1 |
| Security | 6 | 30 | 7 |
| Security | 8 | 22 | 5 |
| Security | 10 | 18 | 3 |
| Security | 12 | 15 | 3 |
| new_metr | 48 | 162 | 40 |
| new_metr | 96 | 81 | 19 |
| new_metr | 192 | 40 | 9 |
| new_metr | 336 | 23 | 5 |
| Total | 12 configurations | 492 | 110 |

These are configuration-window memberships, not necessarily distinct forecast origins across horizons. new_metr has irregular observation timestamps; its horizons are observation steps and must not automatically be described as hours. Health has conflicting duplicate dates and continuity breaks handled by the loader. Current target-contract mapping statuses are Health=conflict, Security=unknown, new_metr=unknown. The paper's descriptions do not by themselves resolve the local source mappings.

The original repository loader's standard sliding-window length is `len(data_x)-seq_len-pred_len+1`; the local HIC adapter advances by a whole horizon and applies additional safety/text filters. Therefore 110 is the local development inventory, not the original paper's count of all sliding-window predictions.

## Conditions for the next experiment

1. Continue using the existing `agent_test_6`, preserving its current payload/vector snapshot and hashes before any new writes. Preserve Test 3 as read-only. The summary records 608 points; this audit did not perform a live Qdrant count.
2. Audit prior saved-output exposure for the 110 candidate windows before attaching any novelty claim. They remain development evaluation candidates. Outer-test stays sealed.
3. The 575 seed records and 33 additions cover five other families; the three remaining families have no corresponding seed memories. Under existing source/target/unit/entity and temporal eligibility, arbitrary cross-domain retrieval is not permitted. Without eligible memory, explicitly record B=A as no-memory fallback.
4. To evaluate actual HIC transfer in these families, first generate and score eligible earlier **fit** experience in the same collection. The 492 fit memberships are a candidate inventory, not an instruction to run all of them. Choose/freeze the bootstrap policy using input/time metadata before validation scoring, exclude self/same-origin records, and enforce `memory.target_end < query.history_start`.
5. Use Codex fresh isolated semantic tasks and frozen BGE-M3, fixed A, identity-bound raw responses and forecast locks, then reveal only the current case's truth. Append/update current corrections in agent_test_6; preserve prior revisions and negative results locally. Later eligible cases may retrieve previously scored memories.
6. If adopting the latest historical-review convention, allow at most three correction revisions for a degraded case, then retain its terminal valid outcome. Keep first-pass metrics and post-review development metrics separately; no guarantee of improvement.
7. Report each six complete cases, by configuration and family, with MAE/MSE/RMSE and improvement/tie/degradation, no-memory fallbacks and technical failures. This expands the HIC branch experiment; it does not reproduce the trained PatchTST/ETA/AFF full VoT architecture.

Reproduce inventory: `python scripts/audit_hic_configuration_coverage_20261030.py`.

Machine-readable evidence: `docs/context/hic_configuration_coverage_20261030.json`.
