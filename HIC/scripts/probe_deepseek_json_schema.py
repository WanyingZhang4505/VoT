"""Probe exact-length Responses API JSON Schema without using test-case data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LLM import hic

HIC_ROOT = ROOT.parents[2]
OUT = HIC_ROOT / "LLM/evaluations/Agent Test 8/deepseek_test_1_20261101_r2/schema_probe"
ENDPOINT = "https://api.deepseek.com/responses"
HORIZON = 192


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if not hic.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    request_body = {
        "model": "deepseek-flash",
        "input": (
            "Return a JSON object matching the supplied schema. The prediction array "
            "must contain exactly 192 numbers, all equal to 1. Keep reasoning brief."
        ),
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "reasoning": {"effort": "none"},
        "store": False,
        "text": {"format": {
            "type": "json_schema",
            "name": "fixed_horizon_probe",
            "schema": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": HORIZON, "maxItems": HORIZON,
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["prediction", "reasoning"],
                "additionalProperties": False,
            },
        }},
    }
    body = json.dumps(request_body, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    request_sha = _sha(body)
    req = Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {hic.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }, method="POST")
    record = {
        "protocol": "deepseek_responses_json_schema_exact_length_probe_v1",
        "endpoint": ENDPOINT,
        "model": request_body["model"],
        "horizon": HORIZON,
        "request_body_sha256": request_sha,
        "request_body": request_body,
        "api_key_configured": True,
        "case_data_included": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with urlopen(req, timeout=120) as response:
            status = getattr(response, "status", 200)
            raw = response.read()
    except HTTPError as error:
        raw = error.read()
        record.update({"status": "http_error", "http_status": error.code,
                       "response_body_sha256": _sha(raw),
                       "response_body": raw.decode("utf-8", errors="replace")[:4000]})
        _write(OUT / "probe.json", record)
        print(json.dumps({"status": record["status"], "http_status": error.code,
                          "request_body_sha256": request_sha,
                          "response_body_sha256": record["response_body_sha256"]}, indent=2))
        return
    except (URLError, OSError) as error:
        record.update({"status": "transport_error", "error_type": type(error).__name__})
        _write(OUT / "probe.json", record)
        print(json.dumps({"status": record["status"], "error_type": type(error).__name__,
                          "request_body_sha256": request_sha}, indent=2))
        return

    parsed_body = json.loads(raw.decode("utf-8"))
    text_parts = [part.get("text", "") for item in parsed_body.get("output", [])
                  if item.get("type") == "message"
                  for part in item.get("content", []) if part.get("type") == "output_text"]
    content = "".join(text_parts)
    try:
        parsed_content = json.loads(content)
        prediction = parsed_content.get("prediction")
        schema_valid = isinstance(prediction, list) and len(prediction) == HORIZON
        if schema_valid:
            schema_valid = all(isinstance(value, (int, float)) and
                               not isinstance(value, bool) for value in prediction)
    except (json.JSONDecodeError, AttributeError):
        parsed_content = None
        prediction = None
        schema_valid = False
    record.update({
        "status": "completed" if status == 200 else "http_error",
        "http_status": status,
        "response_body_sha256": _sha(raw),
        "response_body": raw.decode("utf-8", errors="replace"),
        "response_status": parsed_body.get("status"),
        "returned_model": parsed_body.get("model"),
        "usage": parsed_body.get("usage"),
        "assistant_content_sha256": _sha(content.encode("utf-8")),
        "assistant_content": content,
        "parsed_prediction_count": len(prediction) if isinstance(prediction, list) else None,
        "exact_length_schema_valid": schema_valid,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write(OUT / "probe.json", record)
    print(json.dumps({"status": record["status"], "http_status": status,
                      "response_status": record["response_status"],
                      "returned_model": record["returned_model"],
                      "parsed_prediction_count": record["parsed_prediction_count"],
                      "exact_length_schema_valid": schema_valid,
                      "request_body_sha256": request_sha,
                      "response_body_sha256": record["response_body_sha256"],
                      "usage": record["usage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
