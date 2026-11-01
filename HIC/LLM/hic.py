import argparse
import os, json, yaml
import math
import warnings
from pathlib import Path
from dotenv import load_dotenv

try:
    from .deepseek_client import call_chat_completion
except ImportError:  # Support running this file directly as `python LLM/hic.py`.
    from deepseek_client import call_chat_completion

try:
    from .sample_data import MAX_TRAIN_CASES, load_socialgood_cases
except ImportError:  # Support running this file directly as `python LLM/hic.py`.
    from sample_data import MAX_TRAIN_CASES, load_socialgood_cases

# ================ API and Model Configuration ====================
env_path = Path(__file__).resolve().parent.parent / 'config'/ 'api.env'
model_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'model.yaml'
load_dotenv(env_path)  # write config/api.env into os.environ

with open(model_cfg_path) as f:
    mdl_cfg = yaml.safe_load(f)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = mdl_cfg["model"]

# ================ LLM + Embedding calls =========================
SUMMARY_PROMPT_VERSION = "hic_summary_v2_task_context"


def call_llm(
    prompt, temperature=0.3, max_retries=6, attempt_budget=None,
    spend_budget=None, usage_callback=None
):
    """Call DeepSeek Flash in JSON mode and return assistant text."""
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1
    return call_chat_completion(
        prompt,
        api_key=DEEPSEEK_API_KEY,
        model=MODEL,
        temperature=temperature,
        max_tokens=mdl_cfg.get("max_output_tokens", 2048),
        max_retries=max_retries,
        attempt_budget=attempt_budget,
        spend_budget=spend_budget,
        usage_callback=usage_callback,
    )


def call_json(
    prompt, temperature=0.0, max_retries=6, attempt_budget=None,
    spend_budget=None, usage_callback=None,
):
    """Call DeepSeek, parse its JSON object, and optionally persist usage metadata."""
    raw = call_llm(
        prompt,
        temperature=temperature,
        max_retries=max_retries,
        attempt_budget=attempt_budget,
        spend_budget=spend_budget,
        usage_callback=usage_callback,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict) or "_raw" in parsed:
        raise ValueError("DeepSeek response was not a valid JSON object")
    return parsed


LLM_CALL_COUNT = 0

# Historical summaries are embedded locally; DeepSeek remains responsible for
# generating summaries, forecasts, and correction notes.
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_QUERY_INSTRUCTION = ""
BGE_MAX_LENGTH = 8192
BGE_EMBEDDING_DIMENSION = 1024
_embedder = None

def extract_json(text):
    """Strip optional Markdown fences and parse a model-generated JSON value."""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        return {"_raw": text}   # keep raw so you can inspect on parse failure

# ================== dataset context ================================
dataset_description = (
    "Monthly United States unemployment time series in the SocialGood dataset. "
    "OT is the single numeric target series supplied for each month; the CSV has "
    "no demographic subgroup column. Text consists of timestamped economic and "
    "labor-market facts. Treat the supplied observations as evidence, not as proof "
    "of causes or subgroup-level outcomes."
)
pred_len = 12

industry = "socialgoods"

kb_file = (Path(__file__).resolve().parent / "knowledge_base" /
           f"kb_{industry}_deepseek.json")

#  ==================  Prmopt design ================================
def build_template_prompt(dataset_description=None):
    context = dataset_description or globals()["dataset_description"]
    return f"""You are a professional data scientist. Design a compact JSON
template that will later be filled in to summarize each time window of a dataset
for OT-value forecasting.

Dataset description: {context}

Produce a JSON template with EXACTLY these three top-level sections:

1. "dataset_info": a FLAT set of key-value fields (single level, no nesting)
   capturing the basics — e.g. dataset name, domain, target variable (OT), unit,
   frequency, data modalities. Values here are short descriptors.

2. "predictive_factors": grouped factors, EXACTLY two levels deep — the outer key
   is a factor CATEGORY, with category names end with "<CATEGORY> for prediction". 
   The inner value is a short description of what to extract for that category. 
   Use a few meaningful categories (e.g. temporal dynamics, external drivers, domain signals). 
   Do NOT nest deeper than these two levels.

3. "forecast_outlook": a FLAT set of single-level fields for the actual analysis
   to be filled later — e.g. expected direction, key driver, confidence. Keep each
   value a concise one-line insight, not a paragraph.

STRICT RULES:
- Maximum nesting depth is TWO levels anywhere in the JSON.
- "dataset_info" and "forecast_outlook" must be strictly single-level (flat).
- Keep it concise and readable: few fields, each meaningful. No redundant or
  overlapping fields. Favor analytical depth per field over number of fields.
- Output ONLY the JSON template, no commentary.
    """

def build_summary_prompt(
    template,
    series,
    news,
    *,
    dataset_description=None,
    covariates=None,
    news_quality_detail=None,
    canonical_json_inputs=False,
    target=None,
    unit=None,
    frequency=None,
    horizon=None,
):
    context = dataset_description or globals()["dataset_description"]
    if not isinstance(canonical_json_inputs, bool):
        raise ValueError("canonical_json_inputs must be a boolean")
    covariates_json = (
        json.dumps(covariates, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical_json_inputs else json.dumps(covariates, ensure_ascii=False)
    ) if covariates else None
    if news_quality_detail is not None and not isinstance(news_quality_detail, dict):
        raise ValueError("news_quality_detail must be a JSON object")
    quality_json = (
        json.dumps(
            news_quality_detail,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if canonical_json_inputs
        else json.dumps(news_quality_detail, ensure_ascii=False, indent=2, allow_nan=False)
    ) if news_quality_detail is not None else "not supplied"
    task_contract = build_summary_task_contract(
        target=target, unit=unit, frequency=frequency, horizon=horizon,
    )
    task_context = ""
    if task_contract is not None:
        # The task contract is authoritative over generic/stale template metadata.
        # Keep the caller's template unchanged because it may be shared across cases.
        prompt_template = dict(template)
        dataset_info = template.get("dataset_info")
        if isinstance(dataset_info, dict):
            prompt_dataset_info = dict(dataset_info)
            prompt_dataset_info.update({
                "target_variable": task_contract["target"],
                "target_unit": task_contract["unit"],
                "frequency": task_contract["frequency"],
            })
            prompt_template["dataset_info"] = prompt_dataset_info
    else:
        prompt_template = template
    template_json = (
        json.dumps(prompt_template, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical_json_inputs else json.dumps(prompt_template, indent=2)
    )
    if task_contract is not None:
        task_json = json.dumps(
            task_contract, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        task_context = f"""
FORECAST TASK CONTRACT (metadata available before the forecast; not future observations)
Versioned summary contract: {task_json}
This versioned summary contract is authoritative for target, unit, frequency, and horizon
if it conflicts with the generic reference or template. The template metadata above has
already been normalized to match this contract.
Describe the supplied history and text only. Do not infer or describe future target values.
"""
    return f"""You are a professional data scientist analyzing a specific time
window of data.

TASK
Generate a NEW and UNIQUE analytical summary for this specific time window of the
dataset. Fill in YOUR OWN analysis of THIS window's data.

REFERENCE (context only, do not copy):
{context}

REQUIRED OUTPUT STRUCTURE (follow this JSON structure exactly, with your analysis):
{template_json}

INPUT DATA TO ANALYZE
Time Series (look-back window): {series}
Text Data (news for this window): {news}
{f"Covariate history (same look-back window): {covariates_json}" if covariates_json else ""}
{task_context}

TEXT EVIDENCE QUALITY
Pipeline quality metadata: {quality_json}
The supplied text is not independently source-verified by this pipeline; do not
describe it as verified news. If a text fragment says the target or market is not
directly mentioned, treat that as no direct target evidence. Any adjacent facts
may be used only as indirect context, not as a direct observation or causal proof.

IMPORTANT:
1. Output must be a valid JSON object only.
2. Do NOT copy the reference; create analysis specific to THIS window.
3. Focus on trends, patterns, and event-driven insights from THIS data.
4. Use only facts supported by the supplied series or timestamped text; label inference as inference.
5. Do not invent demographic groups, subgroups, policy actions, or causal relationships.
6. The numeric series is one OT target series; do not claim it is disaggregated by race or other groups.
Respond with ONLY the JSON object."""


def build_summary_task_contract(*, target=None, unit=None, frequency=None, horizon=None):
    """Return stable task metadata for a versioned summary, or None for legacy calls."""
    if all(value is None for value in (target, unit, frequency, horizon)):
        return None
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("versioned summary context requires a positive integer horizon")
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError("versioned summary context requires a non-empty frequency")
    normalized = {}
    for key, value, fallback in (
        ("target", target, "unspecified"),
        ("unit", unit, "unspecified"),
    ):
        if value is None:
            normalized[key] = fallback
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"versioned summary {key} must be a non-empty string")
        else:
            normalized[key] = " ".join(value.split())
    return {
        "version": SUMMARY_PROMPT_VERSION,
        **normalized,
        "frequency": " ".join(frequency.split()).lower(),
        "horizon": horizon,
    }


def forecast_metrics(prediction, truth):
    """Compute numeric forecast errors on one aligned, non-empty target window."""
    predicted = [float(value) for value in prediction]
    observed = [float(value) for value in truth]
    if not observed or len(predicted) != len(observed):
        raise ValueError("prediction and truth must have the same non-zero length")
    if not all(math.isfinite(value) for value in predicted + observed):
        raise ValueError("prediction and truth must contain only finite numbers")
    errors = [pred - actual for pred, actual in zip(predicted, observed)]
    return {
        "n": len(observed),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": sum(error * error for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def persistence_forecast(history, horizon):
    """Repeat the latest observed value as a transparent baseline forecast."""
    if not history or horizon <= 0:
        raise ValueError("history must be non-empty and horizon must be positive")
    return [float(history[-1])] * horizon


def build_reasoning_prompt(
    summary,
    series,
    correction=None,
    retrieval_policy="current",
    covariates=None,
    dataset_description=None,
    horizon=None,
    frequency=None,
    canonical_json_inputs=False,
):
    if retrieval_policy not in {"current", "relevance_gated"}:
        raise ValueError("retrieval_policy must be 'current' or 'relevance_gated'")
    forecast_horizon = pred_len if horizon is None else horizon
    if not isinstance(forecast_horizon, int) or isinstance(forecast_horizon, bool) or forecast_horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(canonical_json_inputs, bool):
        raise ValueError("canonical_json_inputs must be a boolean")
    json_kwargs = (
        {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":"), "allow_nan": False}
        if canonical_json_inputs else {}
    )
    step_label = frequency or "monthly"
    if step_label == "irregular":
        step_label = "observations with irregular time intervals"
        horizon_label = f"{forecast_horizon} {step_label}"
    else:
        horizon_label = f"{forecast_horizon} {step_label} steps"
    p = f""" You are a quantative analyst for multi-variate time series.
    - Prediction Length: {forecast_horizon}
    Use only the supplied numeric history and evidence in SUMMARY. Do not invent
    demographic breakdowns, events, or causal explanations. Return exactly
    {forecast_horizon} numeric predictions for the next {horizon_label}.
    SUMMARY:{json.dumps(summary, **json_kwargs)}
    HISTORY:{series}
    """
    if dataset_description:
        p += f"\n    DATASET CONTEXT:{dataset_description}\n"
    if covariates:
        p += f"\n    COVARIATE HISTORY:{json.dumps(covariates, **json_kwargs)}\n"
    if correction:
        if retrieval_policy == "current":
            p += f"""
        Lessons learned from a past reasoning mistaks:
        {correction}
        """
        else:
            p += f"""
        Retrieved correction from a past case (it may not apply):
        {correction}

        Check whether this past case applies to the current summary and numeric
        history before using its lesson. Use only a matching error pattern; if the
        evidence or trend differs, ignore the lesson. Do not copy its previous
        numeric predictions. Base the forecast on the current history and evidence.
        """
    p += f"""
    OUTPUT JSON exactly:
    {{"prediction": [ {forecast_horizon} numbers ], "reasoning": "brief explanation"}}
    """
    return p


def build_correction_prompt(
    summary,
    series,
    truth,
    pred,
    reasoning,
    *,
    covariates=None,
    dataset_description=None,
):
    p = f"""
    You are an expert in improving reasoning by learning from outcomes.

    SUMMARY:{json.dumps(summary)}
    HISTORY VALUE:{series}
    {f"COVARIATE HISTORY:{json.dumps(covariates, ensure_ascii=False)}" if covariates else ""}
    {f"DATASET CONTEXT:{dataset_description}" if dataset_description else ""}
    GROUND THUTH VALUE:{truth}
    ORIGINAL PREDICTION:{pred}
    ORIGINAL REASONING:{reasoning}

    TASK: Explain the error in reasoning that deviates the prediction from the ground truth values. 
    DO NOT repeat the existing information. Explore the potential indicators to improve the prediciton
    acuracy and improve the reasoning.

    OUTPUT JSON FORMAT: 
    {{"Improved Reasoning": "...", "Key Insights": "Brief description in summarizing the error in previous reasoning."}}
    """
    return p


def _ordered_fields(fields, preferred_keys):
    if not isinstance(fields, dict):
        return []
    ordered = [key for key in preferred_keys if key in fields]
    ordered.extend(sorted(key for key in fields if key not in preferred_keys))
    return [(key, fields[key]) for key in ordered]


def _summary_value_to_text(value):
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def summary_to_embedding_text(summary):
    """Serialize summaries deterministically, prioritizing case-specific signals."""
    if isinstance(summary, str):
        return " ".join(summary.split())
    if not isinstance(summary, dict):
        return _summary_value_to_text(summary)

    section_orders = (
        ("predictive_factors", (
            "Temporal Dynamics for prediction",
            "External Drivers for prediction",
            "Domain Signals for prediction",
            "Textual Signals for prediction",
        )),
        ("forecast_outlook", (
            "expected_direction", "key_driver", "uncertainty", "confidence", "horizon",
        )),
        ("dataset_info", (
            "domain", "target_variable", "unit", "frequency", "geography",
        )),
    )
    lines = []
    for section, preferred_keys in section_orders:
        for key, value in _ordered_fields(summary.get(section), preferred_keys):
            if value is None or value == "":
                continue
            lines.append(f"{key.replace('_', ' ')}: {_summary_value_to_text(value)}")
    if lines:
        return "\n".join(lines)
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LocalBgeEmbedder:
    """Lazy local inference wrapper for the active BGE-M3 sentence encoder."""

    def __init__(self, model_name=BGE_MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        configured_limits = [
            getattr(self.tokenizer, "model_max_length", None),
            getattr(self.model.config, "max_position_embeddings", None),
        ]
        usable_limits = [
            int(limit)
            for limit in configured_limits
            if isinstance(limit, (int, float)) and 0 < limit < 10_000_000
        ]
        if not usable_limits:
            raise RuntimeError(f"Could not determine the context limit for {model_name}.")
        self.max_length = min(BGE_MAX_LENGTH, *usable_limits)

    def encode(self, texts, is_query=False):
        if not texts:
            return []
        prepared = [
            BGE_QUERY_INSTRUCTION + text if is_query else text
            for text in texts
        ]
        untruncated = self.tokenizer(
            prepared, add_special_tokens=True, truncation=False
        )["input_ids"]
        truncated_count = sum(len(tokens) > self.max_length for tokens in untruncated)
        if truncated_count:
            warnings.warn(
                f"{truncated_count} HIC embedding input(s) exceed {self.max_length} "
                "tokens and will be truncated; case-specific signals are serialized "
                "before static dataset metadata.",
                RuntimeWarning,
                stacklevel=2,
            )

        encoded = self.tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model(**encoded)
            vectors = self.torch.nn.functional.normalize(
                output.last_hidden_state[:, 0], p=2, dim=1
            )

        if vectors.ndim != 2 or vectors.shape[1] != BGE_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"{self.model_name} returned {tuple(vectors.shape)}; expected "
                f"(batch, {BGE_EMBEDDING_DIMENSION})."
            )
        if not self.torch.isfinite(vectors).all():
            raise RuntimeError(f"{self.model_name} returned a non-finite embedding.")
        return vectors.cpu().tolist()


def get_embedder():
    """Load BGE on first use and reuse it for subsequent HIC retrievals."""
    global _embedder
    if _embedder is None:
        _embedder = LocalBgeEmbedder()
    return _embedder


def _normalize_stored_embedding(vector, label):
    if not isinstance(vector, (list, tuple)) or len(vector) != BGE_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Knowledge-base embedding {label} must have {BGE_EMBEDDING_DIMENSION} dimensions."
        )
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError):
        raise ValueError(f"Knowledge-base embedding {label} must contain numeric values.") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Knowledge-base embedding {label} contains non-finite values.")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"Knowledge-base embedding {label} must have a non-zero finite norm.")
    return [value / norm for value in values]

# ======================= Get template =================================
# template folder with all industry datasets
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"

def get_template(dataset_name, dataset_description=None):
    """First try to read LLM/template/<dataset_name>.json
    If file does not exist / empty / failed to parse, then use LLM to generate"""
    tpl_path = TEMPLATE_DIR / f"{dataset_name}.json"

    if tpl_path.exists() and tpl_path.stat().st_size > 0:
        try:
            with open(tpl_path) as f:
                tpl = json.load(f)
            if tpl:   # ensure parse successfully and content is non-empty (non-empty dict/list）
                print(f"STEP 1: loaded template from {tpl_path}")
                return tpl
            print(f"STEP 1: {tpl_path} is empty, use LLM to generate ...")
        except json.JSONDecodeError as e:
            print(f"STEP 1: {tpl_path} parse failed {e}), use LLM to generate ...")
    else:
        print(f"STEP 1: Can't find {tpl_path}, use LLM to generate ...")

    return extract_json(call_llm(build_template_prompt(dataset_description)))


# ======================= Generate Prompt =================================
def summarize(template, series, news, *, dataset_description=None, covariates=None):
    return extract_json(call_llm(build_summary_prompt(
        template, series, news,
        dataset_description=dataset_description,
        covariates=covariates,
    )))

def reason(
    summary, series, correction=None, *, dataset_description=None, covariates=None
):
    return extract_json(call_llm(build_reasoning_prompt(
        summary, series, correction,
        dataset_description=dataset_description,
        covariates=covariates,
    )))

def correct(
    summary, series, truth, pred, reasoning, *,
    dataset_description=None, covariates=None,
):
    return extract_json(call_llm(build_correction_prompt(
        summary, series, truth, pred, reasoning,
        dataset_description=dataset_description,
        covariates=covariates,
    )))


def select_similar_case(query_summary, knowledge_base):
    if not knowledge_base:
        raise ValueError("Cannot retrieve a case from an empty knowledge base.")

    for index, record in enumerate(knowledge_base):
        if not isinstance(record, dict) or "summary" not in record or "correction" not in record:
            raise ValueError(
                f"Knowledge-base entry {index} must contain 'summary' and 'correction'."
            )

    embedder = get_embedder()
    query_vectors = embedder.encode(
        [summary_to_embedding_text(query_summary)], is_query=True
    )
    if len(query_vectors) != 1:
        raise RuntimeError("BGE must return exactly one embedding for the query summary.")
    query_vector = _normalize_stored_embedding(query_vectors[0], "query")

    pending_indices = [
        index for index, record in enumerate(knowledge_base)
        if record.get("embedding_model") != BGE_MODEL_NAME or "embedding" not in record
    ]
    if pending_indices:
        passage_texts = [
            summary_to_embedding_text(knowledge_base[index]["summary"])
            for index in pending_indices
        ]
        passage_vectors = embedder.encode(passage_texts, is_query=False)
        if len(passage_vectors) != len(pending_indices):
            raise RuntimeError("BGE returned a different number of passage vectors than inputs.")
        for index, vector in zip(pending_indices, passage_vectors):
            knowledge_base[index]["embedding"] = _normalize_stored_embedding(vector, index)
            knowledge_base[index]["embedding_model"] = BGE_MODEL_NAME

    similarities = []
    for index, record in enumerate(knowledge_base):
        vector = _normalize_stored_embedding(record["embedding"], index)
        similarities.append(sum(left * right for left, right in zip(query_vector, vector)))

    selected_index = max(range(len(similarities)), key=similarities.__getitem__)
    metadata = {
        "method": "bge_m3_cosine",
        "selected_index": selected_index,
        "similarity": similarities[selected_index],
    }
    return knowledge_base[selected_index], metadata

# ======================= Build Knowledge Base ============================
def build_knowledge_base(train_samples, template, append=True, output_path=None):
    """
    input: train_samepls -> [{"train_sample":[...], "series":[...], "truth":[...], "news": "..."}, ...]
    output: kb -> [{"summary": {...}, "correction": "...", "embedding": [...]}, ...]
    """
    destination = Path(output_path) if output_path is not None else kb_file
    if append and destination.exists():
        with destination.open(encoding="utf-8") as handle:
            kb = json.load(handle)
    else:
        kb = []

    for i, s in enumerate(train_samples):
        summary = summarize(template, s["series"], s["news"])
        init_r = reason(summary, s["series"])
        corr_r = correct(summary, s['series'], s['truth'], init_r.get("prediction"), init_r.get("reasoning"))
        kb.append(
            {
                "summary": summary,
                "correction": corr_r.get("Key Insights") or corr_r.get("Improved Reasoning") or str(corr_r),
            }
        )
        print(f"Create KB entry {i+1} / {len(train_samples)}")

    pending_entries = [
        entry for entry in kb
        if entry.get("embedding_model") != BGE_MODEL_NAME or "embedding" not in entry
    ]
    if pending_entries:
        passage_texts = [
            summary_to_embedding_text(entry["summary"])
            for entry in pending_entries
        ]
        vectors = get_embedder().encode(passage_texts, is_query=False)
        if len(vectors) != len(pending_entries):
            raise RuntimeError("BGE returned a different number of passage vectors than inputs.")
        for index, (entry, vector) in enumerate(zip(pending_entries, vectors)):
            entry["embedding"] = _normalize_stored_embedding(vector, index)
            entry["embedding_model"] = BGE_MODEL_NAME

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"Saved KB -> {destination} {len(train_samples)} entries")
    return kb

# ======================= hic prediction =======================
def hic_predict(kb, template, series, news):
    summary = summarize(template, series, news)
    best, retrieval = select_similar_case(summary, kb)
    print(
        f"  retrieved case {retrieval['selected_index']} via local BGE "
        f"(cosine={retrieval['similarity']:.4f})"
    )

    return reason(summary, series, best["correction"]), retrieval

# ======================= RUN =======================
def run(train_count=10):
    global LLM_CALL_COUNT
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before running HIC.")
    if not 1 <= train_count <= MAX_TRAIN_CASES:
        raise ValueError(f"train_count must be between 1 and {MAX_TRAIN_CASES}")

    LLM_CALL_COUNT = 0
    train_samples, test_sample, test_truth = load_socialgood_cases(train_count=train_count)
    print(f"[model={MODEL} | retrieval=local {BGE_MODEL_NAME} cosine similarity]\n")
    print(f"[SocialGood | train_cases={len(train_samples)} | history={8} months | horizon={pred_len} months]")
    template = get_template(industry)
    print(json.dumps(template, indent=2), "\n")

    print("STEP 4-5: building knowledge base from train samples ...")
    # Each requested corpus size has its own reproducible artifact paths.
    output_dir = Path(__file__).resolve().parent
    kb_path = output_dir / "knowledge_base" / f"kb_{industry}_deepseek_{train_count}train.json"
    output_path = output_dir / f"vot_hic_output_{train_count}train.json"
    kb = build_knowledge_base(train_samples, template, append=False, output_path=kb_path)

    print(f"\nSTEP 6: HIC-guided prediction on fixed test sample ({test_sample['case_id']}) ...")
    pred, retrieval = hic_predict(kb, template, test_sample["series"], test_sample["news"])
    if not isinstance(pred, dict) or "prediction" not in pred:
        raise ValueError("HIC returned no prediction array; inspect the saved model response")
    prediction_values = pred["prediction"]
    metrics = forecast_metrics(prediction_values, test_truth)
    baseline_values = persistence_forecast(test_sample["series"], len(test_truth))
    baseline_metrics = forecast_metrics(baseline_values, test_truth)
    mae_improvement = (
        (baseline_metrics["mae"] - metrics["mae"]) / baseline_metrics["mae"]
        if baseline_metrics["mae"] else None
    )
    print(json.dumps(pred, indent=2))
    result = {
        "dataset": "SocialGood",
        "train_case_ids": [sample["case_id"] for sample in train_samples],
        "kb_size": len(kb),
        "test_case": {
            key: test_sample[key]
            for key in ("case_id", "history_start", "history_end", "series")
        },
        "test_prediction": pred,
        "test_ground_truth": test_truth,
        "evaluation": {
            "metric_note": "Forecast errors are computed on the 12 target months after the 8-month test history; truth was not passed to HIC.",
            "hic": metrics,
            "persistence_baseline": baseline_metrics,
            "mae_relative_improvement_vs_persistence": mae_improvement,
        },
        "retrieval": retrieval,
        "logical_deepseek_calls": LLM_CALL_COUNT,
        "logical_call_note": "Counts call_llm invocations; network retries may make the HTTP attempt count higher.",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"template": template, **result}, handle, indent=2, ensure_ascii=False)
    print(f"\nTest MAE={metrics['mae']:.4f}; RMSE={metrics['rmse']:.4f}")
    print(f"Persistence baseline MAE={baseline_metrics['mae']:.4f}; RMSE={baseline_metrics['rmse']:.4f}")
    print(f"Logical DeepSeek calls: {LLM_CALL_COUNT}; retries are not included")
    print(f"Saved results -> {output_path}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local SocialGood HIC example.")
    parser.add_argument("--train-count", type=int, choices=range(1, MAX_TRAIN_CASES + 1), default=10)
    arguments = parser.parse_args()
    run(train_count=arguments.train_count)
