"""Truth-sealed, single-request DeepSeek adapter for Agent Test 8."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from LLM import hic
from LLM.deepseek_client import API_URL, _safe_http_error_diagnostics


BACKEND = "deepseek_api"
PROMPT_VERSION = "agent_test8_deepseek_paired_v1"
PROVIDER_PROTOCOL = "agent_test8_deepseek_one_pass_direct_nearest_v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


class DeepSeekRequestError(RuntimeError):
    """A failed API attempt with safe diagnostics and its audit record path."""

    def __init__(self, message: str, *, record_path: Path):
        super().__init__(message)
        self.record_path = record_path


class DeepSeekRequestCache:
    """Synchronous cache compatible with ``advance_agent_case``'s invoke API.

    A successful, validated response is reused forever for the same request ID.
    A failed transport or invalid response can be retried only by invoking the
    same ID again; the prompt and generation settings are part of its identity.
    There are no automatic retries or result-driven prompt repairs.
    """

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
        self.max_tokens = int(hic.mdl_cfg.get("max_output_tokens", 2048))
        self.timeout_seconds = float(os.environ.get("AGENT_TEST8_DEEPSEEK_TIMEOUT", "120"))
        if self.timeout_seconds <= 0:
            raise ValueError("AGENT_TEST8_DEEPSEEK_TIMEOUT must be positive")
        self.request_settings = {
            "provider": BACKEND,
            "endpoint": API_URL,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "timeout_seconds": self.timeout_seconds,
            "run_id": run_id,
            "dataset": dataset,
            "horizon": horizon,
            "frequency": frequency,
            "target": target,
            "unit": unit,
            "prompt_version": PROMPT_VERSION,
            "protocol": PROVIDER_PROTOCOL,
        }
        self._paths_by_hash: dict[str, Path] = {}

    def _path(self, call_id: str, request_sha256: str) -> Path:
        return self.output_dir / f"{_sha(call_id + ':' + request_sha256)}.json"

    def _identity(self, call_id: str, prompt: str) -> dict:
        prompt_sha = _sha(prompt)
        identity = {
            "request_settings": self.request_settings,
            "call_id": call_id,
            "prompt_sha256": prompt_sha,
        }
        return {
            "sha256": _sha(_canonical(identity)),
            "prompt_sha256": prompt_sha,
            "request_settings": self.request_settings,
            "backend": BACKEND,
        }

    def _request_body(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _validate_cached_response(record: dict, validator) -> dict | None:
        attempt = record["attempts"][-1]
        if attempt.get("status") != "completed":
            return None
        response = attempt["parsed_response"]
        if validator is not None and validator(response) is False:
            raise ValueError("cached DeepSeek response failed local validation")
        return response

    def invoke(
        self, call_id, prompt, *, call_json, validator=None,
        retry_unresolved=False,
    ):
        del call_json, retry_unresolved
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be non-empty text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if validator is not None and not callable(validator):
            raise ValueError("validator must be callable")
        if not hic.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is missing; configure config/api.env")

        identity = self._identity(call_id, prompt)
        path = self._path(call_id, identity["sha256"])
        body = self._request_body(prompt)
        body_text = _canonical(body)
        body_sha = _sha(body_text)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("request_identity", {}).get("sha256") != identity["sha256"]:
                raise ValueError("DeepSeek request identity does not match its content-addressed cache file")
            cached = self._validate_cached_response(record, validator)
            if cached is not None:
                self._paths_by_hash[identity["sha256"]] = path
                return {"response": cached, "request_identity": identity}
            attempts = record["attempts"]
        else:
            record = {
                "schema_version": 1,
                "backend": BACKEND,
                "call_id": call_id,
                "request_identity": identity,
                "request_body_sha256": body_sha,
                "request_body": body,
                "prompt": prompt,
                "attempts": [],
                "created_utc": _utc(),
            }
            attempts = record["attempts"]

        attempt = {
            "attempt": len(attempts) + 1,
            "started_utc": _utc(),
            "status": "in_flight",
        }
        attempts.append(attempt)
        _atomic_json(path, record)
        request = Request(
            API_URL,
            data=body_text.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {hic.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status_code = getattr(response, "status", 200)
                raw_body = response.read()
        except HTTPError as error:
            attempt.update({
                "status": "http_error", "finished_utc": _utc(),
                "http_status": error.code,
                "provider_diagnostics": _safe_http_error_diagnostics(error),
            })
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekRequestError(
                f"DeepSeek returned HTTP {error.code}; the unchanged request is saved for recovery.",
                record_path=path,
            ) from None
        except (URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            attempt.update({
                "status": "transport_error", "finished_utc": _utc(),
                "error_type": type(reason).__name__,
            })
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekRequestError(
                f"DeepSeek transport failed ({type(reason).__name__}); the unchanged request is saved for recovery.",
                record_path=path,
            ) from None

        attempt["http_status"] = status_code
        attempt["finished_utc"] = _utc()
        attempt["response_body_sha256"] = hashlib.sha256(raw_body).hexdigest()
        attempt["response_body"] = raw_body.decode("utf-8", errors="replace")
        if status_code != 200:
            attempt["status"] = "http_error"
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekRequestError(
                f"DeepSeek returned HTTP {status_code}; the unchanged request is saved for recovery.",
                record_path=path,
            )
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            choice = payload["choices"][0]
            message = choice["message"]
            raw_content = message["content"]
            finish_reason = choice.get("finish_reason")
            if not isinstance(raw_content, str) or not raw_content.strip():
                raise ValueError("assistant content is empty")
            parsed = hic.extract_json(raw_content)
            if not isinstance(parsed, dict) or "_raw" in parsed:
                raise ValueError("assistant content is not one JSON object")
            if finish_reason not in {"stop", None}:
                raise ValueError(f"finish_reason={finish_reason}")
            if validator is not None and validator(parsed) is False:
                raise ValueError("response failed stage-specific validation")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError,
                TypeError, ValueError) as error:
            attempt.update({
                "status": "validation_error", "validation_error": str(error)[:240],
                "raw_assistant_content": (raw_content if "raw_content" in locals()
                                           and isinstance(raw_content, str) else None),
            })
            if isinstance(attempt["raw_assistant_content"], str):
                attempt["raw_assistant_content_sha256"] = _sha(attempt["raw_assistant_content"])
            _atomic_json(path, record)
            self._paths_by_hash[identity["sha256"]] = path
            raise DeepSeekRequestError(
                f"DeepSeek response failed JSON/schema validation ({type(error).__name__}); the unchanged request is saved for recovery.",
                record_path=path,
            ) from None

        attempt.update({
            "status": "completed",
            "raw_assistant_content": raw_content,
            "raw_assistant_content_sha256": _sha(raw_content),
            "parsed_response": parsed,
            "finish_reason": finish_reason,
            "returned_model": payload.get("model"),
            "usage": payload.get("usage"),
            "system_fingerprint": payload.get("system_fingerprint"),
        })
        _atomic_json(path, record)
        self._paths_by_hash[identity["sha256"]] = path
        return {"response": parsed, "request_identity": identity}

    def mark_persisted(self, request_sha256):
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
        path = self._path(call_id, request_sha256)
        return json.loads(path.read_text(encoding="utf-8"))


def preflight_configuration() -> dict:
    """Return a safe configuration summary without printing or transmitting the key."""
    return {
        "backend": BACKEND,
        "api_key_configured": bool(hic.DEEPSEEK_API_KEY),
        "endpoint": API_URL,
        "model": str(hic.MODEL),
        "temperature": float(hic.mdl_cfg.get("temperature", 0.7)),
        "max_tokens": int(hic.mdl_cfg.get("max_output_tokens", 2048)),
        "thinking": "disabled",
        "response_format": "json_object",
    }
