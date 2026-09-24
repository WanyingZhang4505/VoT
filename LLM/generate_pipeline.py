import os, json, requests, yaml
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# 定位到 config/api.env（相对本文件，LLM/ 的上一级再进 config/）
env_path = Path(__file__).resolve().parent.parent / "config" / "api.env"
load_dotenv(env_path)       # load API into environment, for client to use

model_cfg_path = Path(__file__).resolve().parent.parent / "config" / "model.yaml"
with open(model_cfg_path) as f:
    mdl_cfg = yaml.safe_load(f)     # convert yaml into a dict

"""
VoT news-structuring pipeline — Gemini version.

Same three-step pipeline as before (template -> summary -> reasoning), rebuilt
from VoT Appendix E, but the LLM call now uses the Google Gemini REST API.

HOW TO RUN
  1. pip install requests
  2. get a Gemini API key from https://aistudio.google.com/apikey
  3. set it:   export GEMINI_API_KEY="your-key"
  4. confirm your model name in the console (see MODEL note below)
  5. python vot_news_pipeline_gemini.py
"""


# ============================================================
# 0. GEMINI API CONFIG
# ============================================================
API_KEY = os.environ.get("GEMINI_API_KEY", "PUT-YOUR-KEY-HERE")

# NOTE ON MODEL NAME:
# There is no "Gemini Flash 3.5". Valid Flash models are e.g.:
#   "gemini-2.5-flash"   (newer, stronger reasoning)
#   "gemini-2.0-flash"   (fast, cheap)
# Check the exact name in https://aistudio.google.com  and put it here.
MODEL = mdl_cfg["model"]

# Gemini REST endpoint (v1beta generateContent)
def _url():
    return (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{MODEL}:generateContent?key={API_KEY}")

def call_llm(prompt, temperature=0.3):
    """One Gemini call. Returns the text content."""
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    r = requests.post(_url(), headers=headers, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Gemini nests the text under candidates[0].content.parts[0].text
    return data["candidates"][0]["content"]["parts"][0]["text"]

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

# ============================================================
# INPUT — a real example shaped like Appendix E (SocialGood / unemployment)
# Replace raw_news and series with your own to try other inputs.
# ============================================================
# dataset_name 决定去 LLM/template/<dataset_name>.json 找已有模板
dataset_name = "socialgood"
dataset_description = ("Monthly unemployment statistics for the United States, "
                       "disaggregated by race. OT value = unemployment rate. "
                       "Text data = economic news and policy reports.")

raw_news = ("Following the September 11 attacks, economists warn of prolonged "
            "labor-market disruption. Several sectors announced layoffs, though "
            "policymakers expect gradual stabilization over the coming months.")

series = [4.9, 4.7, 5.0, 5.3, 5.4, 6.3, 6.1, 6.1]   # Appendix E #611 OT values
pred_len = 12

# ============================================================
# STEP 1 — TEMPLATE GENERATION (Appendix E.1)
# ============================================================
template_prompt = f"""You are a professional data scientist. Design a compact JSON
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
- Output ONLY the JSON template, no commentary."""

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

    return extract_json(call_llm(template_prompt))

# ============================================================
# STEP 2 — SUMMARY (Appendix E.2)
# ============================================================
def build_summary_prompt(template):
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
Text Data (news for this window): {raw_news}

IMPORTANT:
1. Output must be a valid JSON object only.
2. Do NOT copy the reference; create analysis specific to THIS window.
3. Focus on trends, patterns, and event-driven insights from THIS data.
Respond with ONLY the JSON object."""

# ============================================================
# STEP 3 — REASONING (Appendix E.3)
# ============================================================
def build_reasoning_prompt(summary):
    return f"""You are a quantitative analyst specializing in multimodal time
series forecasting.
- Prediction Length: {pred_len}

TASK: Predict the next {pred_len} values using the textual summary and the window.

ANALYSIS:
1. Textual Intelligence: extract insights from the summary.
2. Numerical Patterns: identify trends and cycles in the series.
3. Domain Knowledge: apply sector understanding.

SUMMARY: {json.dumps(summary)}
SERIES: {series}

OUTPUT JSON exactly:
{{"Prediction": [ {pred_len} numbers ],
  "Reasoning": "brief explanation of key factors"}}"""

# ============================================================
# RUN
# ============================================================
def run():
    if API_KEY == "PUT-YOUR-KEY-HERE":
        print("!! Set GEMINI_API_KEY (env var) or edit API_KEY in the file first.")
        print("   Also confirm MODEL matches a real Gemini model in your console.")
        print("\n--- STEP 1 template_prompt (preview) ---\n", template_prompt[:400], "...\n")
        return

    print(f"[using model: {MODEL}]\n")
    template = get_template(dataset_name)
    print(json.dumps(template, indent=2), "\n")

    print("STEP 2: filling template from news+series -> structured summary ...")
    summary = extract_json(call_llm(build_summary_prompt(template)))
    print(json.dumps(summary, indent=2)[:1000], "\n")

    print("STEP 3: reasoning -> numeric prediction ...")
    pred = extract_json(call_llm(build_reasoning_prompt(summary)))
    print(json.dumps(pred, indent=2), "\n")

    with open("vot_pipeline_output.json", "w") as f:
        json.dump({"template": template, "summary": summary, "prediction": pred},
                  f, indent=2)
    print("saved -> vot_pipeline_output.json")

if __name__ == "__main__":
    run()