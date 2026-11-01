"""DeepSeek strict function-call adapter for exact-horizon forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from LLM import hic
from LLM.deepseek_client import _safe_http_error_diagnostics


BACKEND = "deepseek_api_strict_function_call"
PROVIDER_PROTOCOL = "agent_test8_deepseek_strict_tool_forecast_v1"
PROMPT_VERSION = "agent_test8_deepseek_indexed_forecast_fields_v1"
STRICT_API_URL = "https://api.deepseek.com/beta/chat/completions"
JSON_API_URL = "https://api.deepseek.com/chat/completions"
TOOL_NAME = "submit_forecast"
FORECAST_STAGES = {
    "baseline_no_retrieval_paired_v1",
    "full_hic_forecast_paired_v2",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_forecast_tool(horizon: int) -> dict:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    names = [f"step_{index:03d}" for index in range(1, horizon + 1)]
    properties = {name: {"type": "number"} for name in names}
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": True,
            "description": (
                f"Return one forecast value for every numbered future step, "
                f"from step_001 through step_{horizon:03d}."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": names,
                "additionalProperties": False,
            },
        },
    }


def decode_forecast_arguments(arguments: object, horizon: int) -> dict:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(arguments, dict):
        raise ValueError("strict forecast arguments must be an object")
    names = [f"step_{index:03d}" for index in range(1, horizon + 1)]
    if set(arguments) != set(names):
        missing = sorted(set(names) - set(arguments))
        extra = sorted(set(arguments) - set(names))
        raise ValueError(f"forecast fields mismatch (missing={missing[:3]}, extra={extra[:3]})")
    values = []
    for name in names:
        value = arguments[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite numeric scalar")
        values.append(float(value))
    return {"prediction": values, "reasoning": ""}


def make_strict_forecast_prompt(prompt: str, horizon: int) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("forecast prompt must be non-empty text")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    marker = re.search(
        rf"Return exactly\s+{horizon}\s+finite numeric\s+predictions and "
        r"concise reasoning in this JSON schema:", prompt,
    )
    start = marker.start() if marker else -1
    boundary = prompt.find("\n\nFORECAST-TIME INPUT:", start) if marker else -1
    if not marker or boundary < 0:
        raise ValueError("forecast prompt did not match the frozen HIC prompt template")
    instructions = (
        f"Call `{TOOL_NAME}` exactly once. Supply one finite numeric value for "
        f"each required field step_001 through step_{horizon:03d}, in chronological "
        "order. Each field is one forecast step; do not omit or add fields. "
        "Do not return a prose answer or a numeric array."
    )
    return prompt[:start] + instructions + prompt[boundary:]


class DeepSeekStrictRequestError(RuntimeError):
    def __init__(self, message: str, *, record_path: Path):
        super().__init__(message)
        self.record_path = record_path


class DeepSeekStrictRequestCache:
    """Truth-sealed cache; summaries use JSON mode, A/B forecasts use strict tools."""

    def __init__(
        self, output_dir: Path, *, run_id: str, dataset: str, horizon: int,
        frequency: str, target: str, unit: str,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.dataset = dataset
        self.horizon = horizon
        self.frequency = frequency
        self.target = target
        self.unit = unit
        self.model = str(hic.MODEL)
        self.temperature = float(hic.mdl_cfg.get("temperature", 0.7))
        self.max_tokens = max(4096, int(hic.mdl_cfg.get("max_output_tokens", 2048)))
        self.timeout_seconds = float(os.environ.get("AGENT_TEST8_DEEPSEEK_TIMEOUT", "120"))
        if self.timeout_seconds <= 0:
            raise ValueError("AGENT_TEST8_DEEPSEEK_TIMEOUT must be positive")
        self._paths_by_hash: dict[str, Path] = {}

    def generation_config(self) -> dict:
        return {
            "provider": BACKEND,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": "disabled",
            "summary_format": "json_object",
            "forecast_format": "strict_function_call_fixed_fields",
            "forecast_schema_sha256": _sha(_canonical(
                build_forecast_tool(self.horizon)
            )),
            "endpoint": STRICT_API_URL,
        }

    def _request_settings(self, *, strict: bool) -> dict:
        return {
            "provider": BACKEND,
            "endpoint": STRICT_API_URL if strict else JSON_API_URL,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "forecast_format": "strict_function_call_fixed_fields" if strict else None,
            "summary_format": "json_object" if not strict else None,
            "schema_version": "fixed_numbered_fields_v1" if strict else None,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "horizon": self.horizon,
            "frequency": self.frequency,
            "target": self.target,
            "unit": self.unit,
            "prompt_version": PROMPT_VERSION,
            "protocol": PROVIDER_PROTOCOL,
        }

    def _identity(self, call_id: str, prompt: str, *, strict: bool) -> dict:
        request_settings = self._request_settings(strict=strict)
        prompt_sha = _sha(prompt)
        identity = {
            "request_settings": request_settings,
            "call_id": call_id,
            "prompt_sha256": prompt_sha,
        }
        return {
            "sha256": _sha(_canonical(identity)),
            "prompt_sha256": prompt_sha,
            "request_settings": request_settings,
            "backend": BACKEND,
        }

    def _request_body(self, prompt: str, *, strict: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
        }
        if strict:
            function_tool = build_forecast_tool(self.horizon)
            body.update({
                "tools": [function_tool],
                "tool_choice": {
                    "type": "function", "function": {"name": TOOL_NAME},
                },
                "parallel_tool_calls": False,
            })
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _parse_response(payload: dict, *, strict: bool, horizon: int,
                        validator) -> tuple[dict, str, str | None]:
        choice = payload["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
        if strict:
            if finish_reason != "tool_calls":
                raise ValueError(f"strict forecast finish_reason={finish_reason}")
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError("expected exactly one strict forecast tool call")
            function = calls[0].get("function", {})
            if function.get("name") != TOOL_NAME:
                raise ValueError("strict forecast called an unexpected tool")
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, str):
                raise ValueError("strict forecast tool arguments are missing")
            arguments = json.loads(raw_arguments)
            response = decode_forecast_arguments(arguments, horizon)
            if validator is not None and validator(response) is False:
                raise ValueError("strict forecast failed local horizon validation")
            return response, raw_arguments, calls[0].get("id")

        raw_content = message.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError("summary assistant content is empty")
        response = hic.extract_json(raw_content)
        if not isinstance(response, dict) or "_raw" in response:
            raise ValueError("summary response is not one JSON object")
        if finish_reason not in {"stop", None}:
            raise ValueError(f"summary finish_reason={finish_reason}")
        if validator is not None and validator(response) is False:
            raise ValueError("summary response failed local validation")
        return response, raw_content, None

    def invoke(self, call_id, prompt, *, call_json, validator=None,
               retry_unresolved=False):
        del call_json, retry_unresolved
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be non-empty text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if not hic.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is missing; configure config/api.env")
        stage = call_id.rsplit(":", 1)[-1]
        strict = stage in FORECAST_STAGES
        request_prompt = make_strict_forecast_prompt(prompt, self.horizon) if strict else prompt
        identity = self._identity(call_id, request_prompt, strict=strict)
        path = self.output_dir / f"{_sha(call_id + ':' + identity['sha256'])}.json"
        body = self._request_body(request_prompt, strict=strict)
        body_text = _canonical(body)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("request_identity", {}).get("sha256") != identity["sha256"]:
                raise ValueError("strict DeepSeek request identity differs from cache path")
            attempt = record["attempts"][-1]
            if attempt.get("status") == "completed":
                response = attempt["parsed_response"]
                if validator is not None and validator(response) is False:
                    raise ValueError("cached response failed local validation")
                self._paths_by_hash[identity["sha256"]] = path
                return {"response": response, "request_identity": identity}
        else:
            record = {
                "schema_version": 1, "backend": BACKEND,
                "call_id": call_id, "request_identity": identity,
                "request_body_sha256": _sha(body_text), "request_body": body,
                "prompt": request_prompt, "attempts": [], "created_utc": _utc(),
            }
        attempt = {"attempt": len(record["attempts"]) + 1,
                   "started_utc": _utc(), "status": "in_flight"}
        record["attempts"].append(attempt)
        _atomic_json(path, record)
        request = Request(
            STRICT_API_URL if strict else JSON_API_URL,
            data=body_text.encode("utf-8"),
            headers={"Authorization": f"Bearer {hic.DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = getattr(response, "status", 200)
                raw_body = response.read()
        except HTTPError as error:
            attempt.update({"status": "http_error", "finished_utc": _utc(),
                            "http_status": error.code,
                            "provider_diagnostics": _safe_http_error_diagnostics(error)})
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekStrictRequestError(
                f"DeepSeek returned HTTP {error.code}; request evidence is saved.",
                record_path=path,
            ) from None
        except (URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            attempt.update({"status": "transport_error", "finished_utc": _utc(),
                            "error_type": type(reason).__name__})
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekStrictRequestError(
                f"DeepSeek transport failed ({type(reason).__name__}); request evidence is saved.",
                record_path=path,
            ) from None
        attempt.update({"http_status": status_code, "finished_utc": _utc(),
                        "response_body_sha256": _sha(raw_body),
                        "response_body": raw_body.decode("utf-8", errors="replace")})
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            response, raw_assistant, tool_call_id = self._parse_response(
                payload, strict=strict, horizon=self.horizon, validator=validator,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError,
                TypeError, ValueError) as error:
            attempt.update({"status": "validation_error", "validation_error": str(error)[:240]})
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekStrictRequestError(
                f"DeepSeek strict response failed local schema validation ({type(error).__name__}); evidence is saved.",
                record_path=path,
            ) from None
        attempt.update({
            "status": "completed", "raw_assistant_content": raw_assistant,
            "raw_assistant_content_sha256": _sha(raw_assistant),
            "parsed_response": response, "finish_reason": payload["choices"][0].get("finish_reason"),
            "tool_call_id": tool_call_id, "returned_model": payload.get("model"),
            "usage": payload.get("usage"),
            "system_fingerprint": payload.get("system_fingerprint"),
        })
        _atomic_json(path, record)
        self._paths_by_hash[identity["sha256"]] = path
        return {"response": response, "request_identity": identity}

    def mark_persisted(self, request_sha256: str) -> bool:
        path = self._paths_by_hash.get(request_sha256)
        if path is None and self.output_dir.is_dir():
            for candidate in self.output_dir.glob("*.json"):
                try:
                    row = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if row.get("request_identity", {}).get("sha256") == request_sha256:
                    path = candidate
                    break
        if path is None or not path.is_file():
            return False
        record = json.loads(path.read_text(encoding="utf-8"))
        record["state_persisted_utc"] = _utc()
        _atomic_json(path, record)
        return True

    def get_record(self, call_id: str, request_sha256: str) -> dict:
        path = self.output_dir / f"{_sha(call_id + ':' + request_sha256)}.json"
        return json.loads(path.read_text(encoding="utf-8"))


def preflight_configuration() -> dict:
    return {
        "backend": BACKEND,
        "api_key_configured": bool(hic.DEEPSEEK_API_KEY),
        "summary_endpoint": JSON_API_URL,
        "forecast_endpoint": STRICT_API_URL,
        "model": str(hic.MODEL),
        "temperature": float(hic.mdl_cfg.get("temperature", 0.7)),
        "max_tokens": max(4096, int(hic.mdl_cfg.get("max_output_tokens", 2048))),
        "thinking": "disabled",
        "forecast_schema": "strict function call; 192 required numbered numeric fields",
        "qdrant_writes": 0,
    }
