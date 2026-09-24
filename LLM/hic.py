import os, json, yaml, requests
from subprocess import call
import numpy as np
from pathlib import Path
from generate_pipeline import call_llm, extract_json, API_KEY, MODEL

# ================ API and Model Configuration ===================
env_path = Path(__file__).resolve().parent.parent / 'config'/ 'api.env'
model_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'model.yaml'

with open(model_cfg_path) as f:
    mdl_cfg = yaml.safe_load(f)

EMBED_MODEL = mdl_cfg["embedding_model"]

# ================ LLM + Embedding calls =========================
def call_emb(txt):
    """call GEMINI embedding models. Return a text outcome."""
    url_ = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EMBED_MODEL}:embedContent?key={API_KEY}")
    r = requests.post(url_, json={"content": {"parts": [{"text": txt}]}}, timeout=120)
    r.raise_for_status()
    return np.array(r.json(["embedding"]["values"]), type=float)

def cosine(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg(b) + 1e-19)
    return cos


# ================== dataset context ================================
dataset_description = ("Monthly unemployment statistics for the United States, "
                       "disaggregated by race. OT value = unemployment rate. "
                       "Text data = economic news and policy reports.")
pred_len = 12

industry = "socialgoods"


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
    {"prediction": [ {pred_len} numbers ], "reasoning": "brief explanation"}
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
# 模板目录：LLM/template/
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"

def get_template(dataset_name):
    """优先读本地模板 LLM/template/<dataset_name>.json
    若文件不存在 / 为空 / 解析失败，则回退到用 LLM 生成。"""
    tpl_path = TEMPLATE_DIR / f"{dataset_name}.json"

    if tpl_path.exists() and tpl_path.stat().st_size > 0:
        try:
            with open(tpl_path) as f:
                tpl = json.load(f)
            if tpl:   # 解析出来且内容非空（非空 dict/list）
                print(f"STEP 1: loaded template from {tpl_path}")
                return tpl
            print(f"STEP 1: {tpl_path} 内容为空，改用 LLM 生成 ...")
        except json.JSONDecodeError as e:
            print(f"STEP 1: {tpl_path} 解析失败（{e}），改用 LLM 生成 ...")
    else:
        print(f"STEP 1: 未找到 {tpl_path}，改用 LLM 生成 ...")

    return extract_json(call_llm(build_template_prompt()))


# ======================= Generate Prompt =================================
def summarize(template, series, news):
    return extract_json(call_llm(build_summary_prompt(template, series, news)))

def reason(summary, series, correction=None):
    return extract_json(call_llm(build_reasoning_prompt(summary, series, correction)))

def correct(summary, series, truth, pred, reasoning):
    return extract_json(call_llm(build_correction_prompt(summary, series, truth, pred, reasoning)))

# ======================= Build Knowledge Base ============================
def build_knowledge_base(train_samples, template):
    """train_samepls: [{"train_sample":[...], "series":[...], "truth":[...], "news": "..."}, ...]"""
    kb = []

    for i, s in enumerate(train_samples):
        summary = summarize(template, s["series"], s["news"])
        init_r = reason(summary, s["series"])
        corr_r = correct(summary, s['series'], s['truth'], init_r.get["prediction"], init_r.get["reasoning"])
        txt_summary = json.dumps(summary)
        kb.append(
            {
                "embedding": call_emb(txt_summary)
            }
        )