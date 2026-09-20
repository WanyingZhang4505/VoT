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
template_prompt = f"""You are a senior data scientist designing a COMPACT extraction schema for OT-value time-series forecasting.

Dataset description: {dataset_description}

Produce a JSON template (a schema of empty/placeholder fields, NOT an analysis) that a downstream model will later fill for each time window. It must guide extraction of forecasting-relevant signals from both the series and the news text.

HARD CONSTRAINTS:
- Maximum nesting depth = 2 (a top-level key may hold a flat object OR a flat list; no deeper).
- No metadata bloat: include a field ONLY if it plausibly changes the forecast. Drop descriptive fluff (data source, units, frequency) unless forecasting-critical.
- Every field name must be self-explanatory; values are short placeholders describing what to put there.

Cover, compactly, these and more potential forecasting essentials:
1. the temporal trend/seasonality to read from the series,
2. the event/impact signal to extract from the news (what happened, direction, and how long it plausibly lasts),
3. the domain relationships worth considering (temporal / causal — skip any that don't apply),
4. a short reasoning cue linking text to the numeric outlook.
5. more potential forecasting essentials

Output ONLY the JSON template."""

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
    print("STEP 1: generating template ...")
    template = extract_json(call_llm(template_prompt))
    print(json.dumps(template, indent=2), "\n")

    # print("STEP 2: filling template from news+series -> structured summary ...")
    # summary = extract_json(call_llm(build_summary_prompt(template)))
    # print(json.dumps(summary, indent=2)[:1000], "\n")

    # print("STEP 3: reasoning -> numeric prediction ...")
    # pred = extract_json(call_llm(build_reasoning_prompt(summary)))
    # print(json.dumps(pred, indent=2), "\n")

    # with open("vot_pipeline_output.json", "w") as f:
    #     json.dump({"template": template, "summary": summary, "prediction": pred},
    #               f, indent=2)
    # print("saved -> vot_pipeline_output.json")

if __name__ == "__main__":
    run()