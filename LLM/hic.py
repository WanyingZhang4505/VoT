import os, json, yaml, requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ================ API and Model Configuration ====================
env_path = Path(__file__).resolve().parent.parent / 'config'/ 'api.env'
model_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'model.yaml'
load_dotenv(env_path)  # write config/api.env into os.environ

with open(model_cfg_path) as f:
    mdl_cfg = yaml.safe_load(f)

API_KEY = os.environ.get("GEMINI_API_KEY", "PUT-YOUR-KEY-HERE")
MODEL = mdl_cfg["model"]
EMBED_MODEL = mdl_cfg["embedding_model"]

# ================ LLM + Embedding calls =========================
# Transient HTTP statuses worth retrying (server overload / rate limit / gateway).
_RETRY_STATUS = {429, 500, 502, 503, 504}

def call_llm(prompt, temperature=0.3, max_retries=6):
    """One Gemini call. Returns the text content.

    Retries transient server errors (e.g. 503 UNAVAILABLE "high demand")
    with exponential backoff, since the API instructs clients to retry later.
    """
    url_ = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{MODEL}:generateContent?key={API_KEY}")

    import time
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    for attempt in range(max_retries):
        r = requests.post(url_, headers=headers, json=body, timeout=120)
        if r.status_code in _RETRY_STATUS and attempt < max_retries - 1:
            wait = min(2 ** attempt, 32)  # 1,2,4,8,16,32s capped
            print(f"  [call_llm] {r.status_code} transient error; retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries}) ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        # Gemini nests the text under candidates[0].content.parts[0].text
        return data["candidates"][0]["content"]["parts"][0]["text"]


def call_emb(txt):
    """call GEMINI embedding models. Return a text outcome."""
    url_ = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EMBED_MODEL}:embedContent?key={API_KEY}")
    r = requests.post(url_, json={"content": {"parts": [{"text": txt}]}}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embedding"]["values"], dtype=float)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-19)

def extract_json(text):
    """Gemini often wraps JSON in ```json fences; strip and parse."""
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

kb_file = Path(__file__).resolve().parent / "knowledge_base" / f"kb_{industry}.json" 

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

# ======================= Build Knowledge Base ============================
def build_knowledge_base(train_samples, template, append=True):
    """
    input: train_samepls -> [{"train_sample":[...], "series":[...], "truth":[...], "news": "..."}, ...]
    output: kb -> [{"embedding": [...], "correction": "..."}, {...}]
    """
    if append and kb_file.exists():
        kb = json.load(open(kb_file))
    else:
        kb = []

    for i, s in enumerate(train_samples):
        summary = summarize(template, s["series"], s["news"])
        init_r = reason(summary, s["series"])
        corr_r = correct(summary, s['series'], s['truth'], init_r.get("prediction"), init_r.get("reasoning"))
        txt_summary = json.dumps(summary)
        kb.append(
            {
                "embedding": call_emb(txt_summary),
                "correction": corr_r.get("Key Insights") or corr_r.get("Improved Reasoning") or str(corr_r)
            }
        )
        print(f"Create KB entry {i+1} / {len(train_samples)}")
    json.dump(kb, open(kb_file, "w"), indent=2)
    print(f"Saved KB -> {kb_file} {len(train_samples)} entries")
    return kb

# ======================= hic prediction =======================
def hic_predict(kb, template, series, news):
    summary = summarize(template, series, news)
    q = call_emb(json.dumps(summary))
    best_sim, best = max(
        ((cosine(q, np.array(kb_record["embedding"])), kb_record) for kb_record in kb),
        key=lambda x: x[0]
    )
    print(f"  retrieved most similar case (cosine={best_sim:.3f})")

    return reason(summary, series, best["correction"]), best_sim

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
    if API_KEY == "PUT-YOUR-KEY-HERE":
        print("!! Set GEMINI_API_KEY first. Structure is ready; here is one prompt:\n")
        print(build_reasoning_prompt({"trend":"up"}, [1,2,3],
              correction="Do not assume smooth recovery after shocks.")[:400])
        return
    print(f"[model={MODEL} | embed={EMBED_MODEL}]\n")
    template = get_template(industry)
    print(json.dumps(template, indent=2), "\n")

    print("STEP 4-5: building knowledge base from train samples ...")
    kb = build_knowledge_base(train_samples, template)

    print("\nSTEP 6: HIC-guided prediction on test sample (#719) ...")
    pred, sim = hic_predict(kb, template, test_sample["series"], test_sample["news"])
    print(json.dumps(pred, indent=2))
 
    json.dump({"template": template, "kb_size": len(kb),
               "test_prediction": pred, "retrieval_cosine": sim},
              open(Path(__file__).resolve().parent / "vot_hic_output.json", "w"), indent=2)
    print("\nsaved -> vot_hic_output.json")

if __name__ == "__main__":
    run()
    