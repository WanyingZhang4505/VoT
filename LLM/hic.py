import os, json, yaml
import math
import warnings
from pathlib import Path
from dotenv import load_dotenv

try:
    from .deepseek_client import call_chat_completion
except ImportError:  # Support running this file directly as `python LLM/hic.py`.
    from deepseek_client import call_chat_completion

# ================ API and Model Configuration ====================
env_path = Path(__file__).resolve().parent.parent / 'config'/ 'api.env'
model_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'model.yaml'
load_dotenv(env_path)  # write config/api.env into os.environ

with open(model_cfg_path) as f:
    mdl_cfg = yaml.safe_load(f)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = mdl_cfg["model"]

# ================ LLM + Embedding calls =========================
def call_llm(prompt, temperature=0.3, max_retries=6):
    """Call DeepSeek Flash in JSON mode and return assistant text."""
    return call_chat_completion(
        prompt,
        api_key=DEEPSEEK_API_KEY,
        model=MODEL,
        temperature=temperature,
        max_tokens=mdl_cfg.get("max_output_tokens", 2048),
        max_retries=max_retries,
    )


# Historical summaries are embedded locally; DeepSeek remains responsible for
# generating summaries, forecasts, and correction notes.
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_MAX_LENGTH = 512
BGE_EMBEDDING_DIMENSION = 384
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
dataset_description = ("Monthly unemployment statistics for the United States, "
                       "disaggregated by race. OT value = unemployment rate. "
                       "Text data = economic news and policy reports.")
pred_len = 12

industry = "socialgoods"

kb_file = (Path(__file__).resolve().parent / "knowledge_base" /
           f"kb_{industry}_deepseek.json")

#  ==================  Prmopt design ================================
def build_template_prompt():
    return f"""You are a professional data scientist. Design a compact JSON
template that will later be filled in to summarize each time window of a dataset
for OT-value forecasting.

Dataset description: {dataset_description}

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

def build_summary_prompt(template, series, news):
    return f"""You are a professional data scientist analyzing a specific time
window of data.

TASK
Generate a NEW and UNIQUE analytical summary for this specific time window of the
dataset. Fill in YOUR OWN analysis of THIS window's data.

REFERENCE (context only, do not copy):
{dataset_description}

REQUIRED OUTPUT STRUCTURE (follow this JSON structure exactly, with your analysis):
{json.dumps(template, indent=2)}

INPUT DATA TO ANALYZE
Time Series (look-back window): {series}
Text Data (news for this window): {news}

IMPORTANT:
1. Output must be a valid JSON object only.
2. Do NOT copy the reference; create analysis specific to THIS window.
3. Focus on trends, patterns, and event-driven insights from THIS data.
Respond with ONLY the JSON object."""


def build_reasoning_prompt(summary, series, correction=None):
    p = f""" You are a quantative analyst for multi-variate time series.
    - Prediction Length: {pred_len}
    SUMMARY:{json.dumps(summary)}
    HISTORY:{series}
    """
    if correction:
        p += f"""
        Lessons learned from a past reasoning mistaks:
        {correction}
        """
    p += f"""
    OUTPUT JSON exactly:
    {{"prediction": [ {pred_len} numbers ], "reasoning": "brief explanation"}}
    """
    return p


def build_correction_prompt(summary, series, truth, pred, reasoning):
    p = f"""
    You are an expert in improving reasoning by learning from outcomes.

    SUMMARY:{json.dumps(summary)}
    HISTORY VALUE:{series}
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
    """Lazy local inference wrapper for the BGE sentence encoder."""

    def __init__(self, model_name=BGE_MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name

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
        truncated_count = sum(len(tokens) > BGE_MAX_LENGTH for tokens in untruncated)
        if truncated_count:
            warnings.warn(
                f"{truncated_count} HIC embedding input(s) exceed {BGE_MAX_LENGTH} "
                "tokens and will be truncated; case-specific signals are serialized "
                "before static dataset metadata.",
                RuntimeWarning,
                stacklevel=2,
            )

        encoded = self.tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=BGE_MAX_LENGTH,
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

def get_template(dataset_name):
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

    return extract_json(call_llm(build_template_prompt()))


# ======================= Generate Prompt =================================
def summarize(template, series, news):
    return extract_json(call_llm(build_summary_prompt(template, series, news)))

def reason(summary, series, correction=None):
    return extract_json(call_llm(build_reasoning_prompt(summary, series, correction)))

def correct(summary, series, truth, pred, reasoning):
    return extract_json(call_llm(build_correction_prompt(summary, series, truth, pred, reasoning)))


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
        "method": "bge_small_cosine",
        "selected_index": selected_index,
        "similarity": similarities[selected_index],
    }
    return knowledge_base[selected_index], metadata

# ======================= Build Knowledge Base ============================
def build_knowledge_base(train_samples, template, append=True):
    """
    input: train_samepls -> [{"train_sample":[...], "series":[...], "truth":[...], "news": "..."}, ...]
    output: kb -> [{"summary": {...}, "correction": "...", "embedding": [...]}, ...]
    """
    if append and kb_file.exists():
        kb = json.load(open(kb_file))
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

    kb_file.parent.mkdir(parents=True, exist_ok=True)
    with open(kb_file, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"Saved KB -> {kb_file} {len(train_samples)} entries")
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

# ======================= TEST =======================
train_samples = [{
    "series": [4.9, 4.7, 5.0, 5.3, 5.4, 6.3, 6.1, 6.1],                 # #611 OT
    "news":   ("Following the September 11 attacks, economists warn of prolonged "
               "labor-market disruption; layoffs announced across sectors."),
    "truth":  [5.7, 5.5, 6.0, 5.9, 5.7, 5.4, 5.3, 5.6, 5.7, 6.5, 6.4, 6.2],  # #611 actual
}]
test_sample = {
    "series": [9.7, 10.6, 10.4, 10.2, 9.5, 9.3, 9.6, 9.7],              # #719 OT
    "news":   ("Amid the 2009-2010 financial crisis aftermath, reports point to a "
               "fragile labor market with slow, uneven recovery."),
}


# ======================= RUN =======================
def run():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY in config/api.env before running HIC.")
    print(f"[model={MODEL} | retrieval=local {BGE_MODEL_NAME} cosine similarity]\n")
    template = get_template(industry)
    print(json.dumps(template, indent=2), "\n")

    print("STEP 4-5: building knowledge base from train samples ...")
    kb = build_knowledge_base(train_samples, template, append=False)

    print("\nSTEP 6: HIC-guided prediction on test sample (#719) ...")
    pred, retrieval = hic_predict(kb, template, test_sample["series"], test_sample["news"])
    print(json.dumps(pred, indent=2))
 
    json.dump({"template": template, "kb_size": len(kb),
               "test_prediction": pred, "retrieval": retrieval},
              open(Path(__file__).resolve().parent / "vot_hic_output.json", "w"), indent=2)
    print("\nsaved -> vot_hic_output.json")

if __name__ == "__main__":
    run()
