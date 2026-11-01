"""Exact-request durable cache for resumable HIC development runs.

Prompts are hashed rather than written to disk. Provider credentials are never
accepted as part of ``request_settings``. A request left in ``request_started``
is treated as ambiguous and needs explicit retry authorization.
"""

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from LLM.deepseek_client import DeepSeekHTTPError
from LLM.deepseek_usage import CNYBudgetBlocked


_SCHEMA_VERSION = 1
_SECRET_PARTS = ("api_key", "authorization", "credential", "password", "secret", "token")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_secret_fields(value, path="request_settings"):
    if isinstance(value, dict):
        for key, nested in value.items():
            name = str(key).lower().replace("-", "_")
            if any(part in name for part in _SECRET_PARTS):
                raise ValueError(f"secret-like field is not allowed in {path}: {key}")
            _reject_secret_fields(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_secret_fields(nested, f"{path}[{index}]")


class RequestCache:
    """Cache validated JSON responses by exact prompt and public request settings."""

    def __init__(self, path, *, request_settings, max_logical_calls, reuse_cache_paths=()):
        if not isinstance(request_settings, dict) or not request_settings:
            raise ValueError("request_settings must be a non-empty dictionary")
        if type(max_logical_calls) is not int or max_logical_calls < 0:
            raise ValueError("max_logical_calls must be a non-negative integer")
        _reject_secret_fields(request_settings)
        self.path = Path(path)
        self.request_settings = json.loads(_canonical(request_settings))
        self.max_logical_calls = max_logical_calls
        self.reuse_cache_paths = self._load_reuse_caches(reuse_cache_paths)
        self.reuse_sources = [source["metadata"] for source in self.reuse_cache_paths]
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError("request cache schema version is not supported")
            if type(state.get("logical_calls")) is not int or state["logical_calls"] < 0:
                raise ValueError("request cache logical call count is invalid")
            if not isinstance(state.get("records"), dict):
                raise ValueError("request cache records are invalid")
            self.state = state
            saved_sources = state.get("reuse_sources", [])
            if saved_sources != self.reuse_sources:
                raise ValueError("request cache reuse sources do not match the current run")
            if type(state.get("reuse_rejected_count", 0)) is not int:
                raise ValueError("request cache reuse rejection count is invalid")
            state.setdefault("reuse_rejected_count", 0)
        else:
            self.state = {
                "schema_version": _SCHEMA_VERSION,
                "logical_calls": 0,
                "records": {},
                "reuse_sources": self.reuse_sources,
                "reuse_rejected_count": 0,
            }

    def _load_reuse_caches(self, paths):
        if paths is None:
            paths = ()
        if isinstance(paths, (str, Path)):
            paths = (paths,)
        if not isinstance(paths, (list, tuple)):
            raise ValueError("reuse_cache_paths must be a path or sequence of paths")

        output_path = self.path.resolve()
        seen = set()
        sources = []
        for value in paths:
            source_path = Path(value).resolve(strict=True)
            if source_path == output_path:
                raise ValueError("a request cache cannot reuse itself")
            if source_path in seen:
                raise ValueError("duplicate request-cache reuse source")
            seen.add(source_path)
            raw = source_path.read_bytes()
            try:
                state = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"reuse request cache is invalid JSON: {source_path}") from error
            if (
                not isinstance(state, dict)
                or state.get("schema_version") != _SCHEMA_VERSION
                or type(state.get("logical_calls")) is not int
                or state["logical_calls"] < 0
                or not isinstance(state.get("records"), dict)
            ):
                raise ValueError(f"reuse request cache has an unsupported structure: {source_path}")

            reusable = {}
            for key, record in state["records"].items():
                if not isinstance(record, dict):
                    raise ValueError(f"reuse request cache contains a malformed record: {source_path}")
                identity = record.get("identity")
                if (
                    not isinstance(identity, dict)
                    or identity.get("sha256") != key
                    or not isinstance(identity.get("prompt_sha256"), str)
                    or not isinstance(identity.get("request_settings"), dict)
                ):
                    raise ValueError(f"reuse request cache identity is invalid: {source_path}")
                if record.get("status") not in {"validated", "persisted"} or "response" not in record:
                    continue
                try:
                    _canonical(record["response"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"reuse request cache response is not finite JSON: {source_path}") from error
                reusable[key] = record

            sources.append({
                "records": reusable,
                "metadata": {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "record_count": len(state["records"]),
                    "reusable_response_count": len(reusable),
                },
            })
        return sources

    @property
    def records(self):
        return self.state["records"]

    @property
    def logical_calls(self):
        return self.state["logical_calls"]

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        preserve_temporary = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                prefix=f".{self.path.stem}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.state, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError as error:
                    if attempt == 4:
                        preserve_temporary = True
                        raise PermissionError(
                            f"could not replace request cache after 5 attempts; "
                            f"complete recovery file retained at {temporary}"
                        ) from error
                    time.sleep(0.1 * (attempt + 1))
        finally:
            if temporary is not None and temporary.exists() and not preserve_temporary:
                temporary.unlink()

    def _identity(self, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty text")
        prompt_hash = _sha256(prompt)
        request_hash = _sha256(_canonical({
            "request_settings": self.request_settings,
            "prompt_sha256": prompt_hash,
        }))
        return {
            "sha256": request_hash,
            "prompt_sha256": prompt_hash,
            "request_settings": self.request_settings,
        }

    @staticmethod
    def _validate(response, validator):
        if validator is None:
            return
        result = validator(response)
        if result is False:
            raise ValueError("invalid response")

    def invoke(self, call_id, prompt, *, call_json, validator=None, retry_unresolved=False):
        """Return an exact cached response or make one explicitly budgeted call."""
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be non-empty text")
        if not callable(call_json):
            raise ValueError("call_json must be callable")
        if validator is not None and not callable(validator):
            raise ValueError("validator must be callable")
        identity = self._identity(prompt)
        key = identity["sha256"]
        record = self.records.get(key)
        if record is not None and record.get("identity") != identity:
            raise ValueError("request cache identity collision or corrupted record")

        if record and record.get("status") in {"validated", "persisted"}:
            try:
                self._validate(record["response"], validator)
            except Exception as error:
                record["status"] = "failed"
                record["validation_error_type"] = type(error).__name__
                self._save()
                raise
            return {"response": record["response"], "request_identity": identity}

        if record is None or "response" not in record:
            for source in self.reuse_cache_paths:
                cached = source["records"].get(key)
                if cached is None:
                    continue
                if cached.get("identity") != identity:
                    raise ValueError("reuse cache identity collision or corrupted record")
                try:
                    self._validate(cached["response"], validator)
                except Exception as error:
                    if record is None:
                        record = {
                            "identity": identity,
                            "status": "planned",
                            "attempts": 0,
                            "call_id_hashes": [],
                            "reuse_rejections": [],
                        }
                        self.records[key] = record
                    rejections = record.setdefault("reuse_rejections", [])
                    source_sha256 = source["metadata"]["sha256"]
                    if not any(
                        item.get("source_cache_sha256") == source_sha256
                        for item in rejections
                        if isinstance(item, dict)
                    ):
                        self.state["reuse_rejected_count"] += 1
                        rejections.append({
                            "source_cache_sha256": source_sha256,
                            "validation_error_type": type(error).__name__,
                        })
                    self._save()
                    continue

                if record is None:
                    record = {
                        "identity": identity,
                        "status": "validated",
                        "attempts": 0,
                        "call_id_hashes": [],
                    }
                    self.records[key] = record
                call_id_hash = _sha256(call_id)
                if call_id_hash not in record["call_id_hashes"]:
                    record["call_id_hashes"].append(call_id_hash)
                record["response"] = cached["response"]
                record["status"] = "validated"
                record["reused_from"] = {
                    "cache_path": source["metadata"]["path"],
                    "cache_sha256": source["metadata"]["sha256"],
                    "record_sha256": key,
                }
                record.pop("validation_error_type", None)
                record.pop("last_error_type", None)
                record.pop("last_http_error", None)
                self._save()
                return {"response": record["response"], "request_identity": identity}

        if record and record.get("status") == "failed" and "response" not in record and not retry_unresolved:
            raise RuntimeError("failed request has no reusable response; explicit retry_unresolved=True is required")
        if record and "response" in record:
            try:
                self._validate(record["response"], validator)
            except Exception as error:
                record["status"] = "failed"
                record["validation_error_type"] = type(error).__name__
                self._save()
                raise
            record["status"] = "validated"
            record.pop("validation_error_type", None)
            self._save()
            return {"response": record["response"], "request_identity": identity}

        if record and record.get("status") == "request_started" and not retry_unresolved:
            raise RuntimeError("request is unresolved; explicit retry_unresolved=True is required")
        if self.logical_calls >= self.max_logical_calls:
            raise RuntimeError(f"logical call budget ({self.max_logical_calls}) exhausted")

        if record is None:
            record = {
                "identity": identity,
                "status": "planned",
                "attempts": 0,
                "call_id_hashes": [],
            }
            self.records[key] = record
        call_id_hash = _sha256(call_id)
        if call_id_hash not in record["call_id_hashes"]:
            record["call_id_hashes"].append(call_id_hash)
        record["status"] = "request_started"
        record["attempts"] += 1
        self.state["logical_calls"] += 1
        self._save()
        try:
            response = call_json(prompt, temperature=0.0)
        except Exception as error:
            record["last_error_type"] = type(error).__name__
            if isinstance(error, CNYBudgetBlocked):
                record["status"] = "not_sent"
                record["last_error_phase"] = "pre_http_budget_guard"
                record.pop("last_http_error", None)
            elif isinstance(error, DeepSeekHTTPError):
                record["last_http_error"] = {
                    "status_code": error.status_code,
                    "provider_diagnostics": error.provider_diagnostics,
                }
            else:
                record.pop("last_http_error", None)
            self._save()
            raise

        try:
            serialized = _canonical(response)
        except (TypeError, ValueError):
            record["status"] = "failed"
            record["last_error_type"] = "NonJsonResponse"
            self._save()
            raise ValueError("provider response must be finite JSON-compatible data") from None
        response = json.loads(serialized)
        record["response"] = response
        record["status"] = "response_received"
        record.pop("last_error_type", None)
        record.pop("last_http_error", None)
        self._save()
        try:
            self._validate(response, validator)
        except Exception as error:
            record["status"] = "failed"
            record["validation_error_type"] = type(error).__name__
            self._save()
            raise
        record["status"] = "validated"
        record.pop("validation_error_type", None)
        self._save()
        return {"response": response, "request_identity": identity}

    def mark_persisted(self, request_sha256):
        """Mark an already validated response as committed to its downstream artifact."""
        record = self.records.get(request_sha256)
        if record is None:
            raise KeyError("request identity is not present in this cache")
        if record.get("status") not in {"validated", "persisted"}:
            raise RuntimeError("only a validated response can be marked persisted")
        record["status"] = "persisted"
        self._save()

    def reuse_summary(self):
        counts = Counter()
        for record in self.records.values():
            source = record.get("reused_from") if isinstance(record, dict) else None
            if isinstance(source, dict):
                counts[source.get("cache_sha256", "unknown")] += 1
        return {
            "source_caches": self.reuse_sources,
            "reused_response_count": sum(counts.values()),
            "by_source": [
                {"sha256": digest, "count": count}
                for digest, count in sorted(counts.items())
            ],
            "rejected_response_count": self.state.get("reuse_rejected_count", 0),
        }
