"""Durable, local handoff queue for Codex-dispatched HIC inference tasks.

The queue deliberately does not make provider requests. A Codex session reads a
pending task, dispatches it to a task agent, validates the returned JSON, and
records the response so the HIC runner can resume deterministically. Context
isolation is controlled by the dispatcher, not enforced by this queue.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable


BACKEND = "codex_task_agent"
SCHEMA_VERSION = 1
_SECRET_KEY_PARTS = ("api_key", "authorization", "credential", "password", "secret", "token")
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?:\bsk-[a-z0-9_-]{20,}\b|\bbearer\s+[a-z0-9._~+/=-]{16,}|"
    r"\b(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[^\s,;]{12,})"
)


def _canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_secret_fields(value, path="metadata") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                raise ValueError(f"secret-like field is not allowed in {path}: {key}")
            _reject_secret_fields(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_secret_fields(nested, f"{path}[{index}]")


def _filesystem_path(path: Path | str) -> str:
    value = os.fspath(path)
    if os.name != "nt":
        return value
    absolute = os.path.abspath(value)
    if absolute.startswith("\\\\?\\") or len(absolute) < 260:
        return value
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def read_utf8_text(path: Path | str) -> str:
    with open(_filesystem_path(path), encoding="utf-8") as handle:
        return handle.read()


def _path_exists(path: Path) -> bool:
    return os.path.exists(_filesystem_path(path))


def _path_is_file(path: Path) -> bool:
    return os.path.isfile(_filesystem_path(path))


def _replace_atomic_path(source: Path, destination: Path) -> None:
    os.replace(_filesystem_path(source), _filesystem_path(destination))


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".task-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for delay in (0.05, 0.1, 0.2, 0.4, 0.8):
            try:
                _replace_atomic_path(temporary, path)
                return
            except PermissionError:
                time.sleep(delay)
        _replace_atomic_path(temporary, path)
    finally:
        if temporary is not None and _path_exists(temporary):
            os.unlink(_filesystem_path(temporary))


class CodexTaskQueue:
    """Store and resume HIC tasks handled by the in-app Codex task agent."""

    _TASK_TYPES = {"summary", "baseline", "correction", "fit_reflection"}
    _STATUSES = {
        "pending", "dispatched", "repair_required", "completed", "persisted", "failed"
    }

    def __init__(self, root, *, max_repairs: int = 1):
        if type(max_repairs) is not int or max_repairs < 0:
            raise ValueError("max_repairs must be a non-negative integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_repairs = max_repairs

    @staticmethod
    def _validate(response, validator):
        if validator is None:
            return
        result = validator(response)
        if result is False:
            raise ValueError("invalid response")

    @staticmethod
    def _task_key(task_id: str) -> str:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        return _sha256_text(task_id.strip())

    def _path(self, task_id: str) -> Path:
        return self.root / f"{self._task_key(task_id)}.json"

    def enqueue(
        self,
        task_id: str,
        *,
        task_type: str,
        prompt: str,
        prompt_version: str,
        output_schema: str,
        metadata=None,
        request_sha256: str | None = None,
    ) -> dict:
        task_hash = self._task_key(task_id)
        if task_type not in self._TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task_type}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if _CREDENTIAL_VALUE.search(prompt):
            raise ValueError("credential-like value is not allowed in the task prompt")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version must be a non-empty string")
        if not isinstance(output_schema, str) or not output_schema.strip():
            raise ValueError("output_schema must be a non-empty string")
        normalized_metadata = {} if metadata is None else metadata
        if not isinstance(normalized_metadata, dict):
            raise ValueError("metadata must be a JSON object")
        _reject_secret_fields(normalized_metadata)

        prompt_sha256 = _sha256_text(prompt)
        identity = {
            "backend": BACKEND,
            "task_type": task_type,
            "prompt_sha256": prompt_sha256,
            "prompt_version": prompt_version,
            "output_schema": output_schema,
            "metadata": normalized_metadata,
            "request_sha256": request_sha256,
        }
        identity_sha256 = _sha256_text(_canonical(identity))
        path = self.root / f"{task_hash}.json"
        if _path_exists(path):
            record = self._read(path)
            if record.get("task_identity_sha256") != identity_sha256:
                raise ValueError("task identity changed for an existing task_id")
            return record

        record = {
            "schema_version": SCHEMA_VERSION,
            "backend": BACKEND,
            "task_id_sha256": task_hash,
            "task_identity_sha256": identity_sha256,
            "request_sha256": request_sha256,
            "task_type": task_type,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "output_schema": output_schema,
            "metadata": json.loads(_canonical(normalized_metadata)),
            "prompt": prompt,
            "status": "pending",
            "dispatch_count": 0,
            "repair_count": 0,
            "deepseek_request_count": 0,
        }
        _atomic_json(path, record)
        return record

    def _read(self, path: Path) -> dict:
        try:
            read_delays = (0.05, 0.1, 0.2, 0.4, 0.8)
            for attempt in range(len(read_delays) + 1):
                try:
                    raw = read_utf8_text(path)
                    break
                except PermissionError:
                    if attempt == len(read_delays):
                        raise
                    time.sleep(read_delays[attempt])
            record = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Codex task record is unreadable: {path.name}") from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("backend") != BACKEND
            or record.get("status") not in self._STATUSES
        ):
            raise ValueError(f"Codex task record has an unsupported structure: {path.name}")
        if "raw_response_text" in record:
            raw_text = record["raw_response_text"]
            if not isinstance(raw_text, str) or not isinstance(record.get("raw_response_sha256"), str):
                raise ValueError("saved raw worker response provenance is malformed")
            if _sha256_text(raw_text) != record["raw_response_sha256"]:
                raise ValueError("saved raw worker response hash mismatch")
            try:
                raw_response = json.loads(raw_text)
                raw_matches = raw_response == record.get("response")
                if not raw_matches and (
                        (record.get("metadata") or {}).get("require_raw_response_provenance") is True
                        and isinstance(raw_response, dict)):
                    provenance = {
                        "task_id": raw_response.get("task_id"),
                        "request_sha256": raw_response.get("request_sha256"),
                        "worker_context_id": raw_response.get("worker_context_id"),
                    }
                    metadata = record.get("metadata") or {}
                    expected_task_id = record.get("task_id")
                    if expected_task_id is None and all(
                            metadata.get(key) is not None
                            for key in ("run_id", "case_id", "horizon", "stage")):
                        expected_task_id = (
                            f"{metadata['run_id']}:{metadata['case_id']}:"
                            f"H{metadata['horizon']}:{metadata['stage']}"
                        )
                    expected = {
                        "task_id": expected_task_id,
                        "request_sha256": record.get("request_sha256"),
                        "worker_context_id": record.get("worker_context_id"),
                    }
                    model_response = {
                        key: value for key, value in raw_response.items()
                        if key not in {"task_id", "request_sha256", "worker_context_id"}
                    }
                    optional_bindings = {
                        "task_identity_sha256": record.get("task_identity_sha256"),
                        "case_id": metadata.get("case_id"),
                        "dataset": metadata.get("dataset"),
                        "horizon": metadata.get("horizon"),
                        "revision_id": metadata.get("memory_version"),
                    }
                    optional_fields_valid = True
                    for key, expected_value in optional_bindings.items():
                        if key in model_response:
                            if expected_value is None or model_response[key] != expected_value:
                                optional_fields_valid = False
                                break
                            model_response.pop(key)
                    raw_matches = (provenance == expected and optional_fields_valid
                                   and model_response == record.get("response"))
                if not raw_matches:
                    raise ValueError("saved raw worker response does not match parsed response")
            except json.JSONDecodeError:
                raise ValueError("saved raw worker response is not valid JSON") from None
            if not isinstance(record.get("response_source"), str) or not record["response_source"].strip():
                raise ValueError("saved raw worker response source is missing")
        return record

    def get(self, task_id: str) -> dict:
        path = self._path(task_id)
        missing_delays = (0.05, 0.1, 0.2)
        for attempt in range(len(missing_delays) + 1):
            if _path_is_file(path):
                break
            if attempt == len(missing_delays):
                raise KeyError("Codex task does not exist")
            time.sleep(missing_delays[attempt])
        return self._read(path)

    def list(self, *, status: str | None = None) -> list[dict]:
        if status is not None and status not in self._STATUSES:
            raise ValueError(f"unsupported task status: {status}")
        records = [self._read(path) for path in self.root.glob("*.json")]
        if status is not None:
            records = [record for record in records if record["status"] == status]
        return sorted(records, key=lambda record: record["task_id_sha256"])

    def handoff_message(self, task_id: str) -> str:
        record = self.get(task_id)
        if record["status"] not in {"pending", "repair_required"}:
            raise RuntimeError("only pending or repair-required tasks can be handed off")
        repair_note = ""
        isolation_note = ""
        if (record.get("metadata") or {}).get("context_isolation_required") is True:
            isolation_note = (
                "This task requires an isolated context. The dispatcher must use a newly "
                "spawned subagent with fork_context=false; if that cannot be guaranteed, "
                "do not answer. The queue records the worker identity but cannot itself "
                "create or verify the context.\n\n"
            )
        if record["status"] == "repair_required":
            repair_note = (
                "The previous response failed schema validation ("
                f"{record.get('validation_error_type', 'invalid response')}). "
                "Regenerate the result once and follow the requested JSON schema exactly.\n\n"
            )
        return (
            "Complete only the task below, using only the evidence included in this task. "
            "Do not use information from other cases or earlier task outputs. The queue "
            "cannot enforce a fresh model context: if this worker has seen other case data, "
            "do not answer and report that context isolation is unavailable. Do not call "
            "external APIs or inspect other task files. Return only one valid JSON object "
            "matching the requested output schema.\n\n"
            + isolation_note
            + repair_note
            + record["prompt"]
        )

    def mark_dispatched(
        self,
        task_id: str,
        *,
        worker_context_id: str | None = None,
        context_isolation_mode: str | None = None,
    ) -> dict:
        path = self._path(task_id)
        record = self._read(path)
        if record["status"] not in {"pending", "repair_required"}:
            raise RuntimeError("only pending or repair-required tasks can be dispatched")
        if record["status"] == "repair_required" and record["repair_count"] > self.max_repairs:
            raise RuntimeError("task repair limit has been reached")
        isolation_required = (
            (record.get("metadata") or {}).get("context_isolation_required") is True
            or bool(record.get("recovery_history"))
        )
        if isolation_required:
            if not isinstance(worker_context_id, str) or not worker_context_id.strip():
                raise ValueError("worker_context_id is required for isolated tasks")
            if context_isolation_mode != "fresh_subagent_no_fork":
                raise ValueError("isolated tasks require context_isolation_mode=fresh_subagent_no_fork")
        elif worker_context_id is not None and (
            not isinstance(worker_context_id, str) or not worker_context_id.strip()
        ):
            raise ValueError("worker_context_id must be non-empty text when provided")
        if worker_context_id is not None:
            for candidate in self.list():
                if candidate["task_id_sha256"] == record["task_id_sha256"]:
                    used_ids = candidate.get("worker_context_ids", [])
                else:
                    used_ids = candidate.get("worker_context_ids", [])
                if worker_context_id in used_ids:
                    raise ValueError("worker_context_id is already assigned to another dispatch")
            record["worker_context_ids"] = [
                *record.get("worker_context_ids", []), worker_context_id
            ]
            record["worker_context_id"] = worker_context_id
        if context_isolation_mode is not None:
            record["context_isolation_mode"] = context_isolation_mode
        record["status"] = "dispatched"
        record["dispatch_count"] += 1
        record.setdefault("dispatch_history", []).append({
            "dispatch_number": record["dispatch_count"],
            "dispatched_utc": datetime.now(timezone.utc).isoformat(),
            "worker_context_id": worker_context_id,
            "context_isolation_mode": context_isolation_mode,
            "task_identity_sha256": record["task_identity_sha256"],
            "request_sha256": record.get("request_sha256"),
        })
        _atomic_json(path, record)
        return record

    def record_response(
        self,
        task_id: str,
        response,
        validator: Callable[[dict], object] | None = None,
        *,
        worker_context_id: str | None = None,
        raw_response_text: str | None = None,
        response_source: str | None = None,
    ) -> dict:
        """Save a response only from the worker assigned to the current dispatch."""
        path = self._path(task_id)
        record = self._read(path)
        if record["status"] != "dispatched":
            raise RuntimeError("task response requires a dispatched task")
        record["task_id"] = task_id
        assigned_worker = record.get("worker_context_id")
        if assigned_worker is not None and worker_context_id != assigned_worker:
            raise ValueError("worker_context_id must match the current dispatched worker")
        if record.get("recovery_history") and assigned_worker is None:
            # Recovered requests need a new worker identity before a response
            # can be accepted; task/request identity alone cannot reject late output.
            raise ValueError("worker_context_id must match the current recovered dispatch")
        if (record.get("metadata") or {}).get("require_raw_response_provenance") is True and (
            not isinstance(raw_response_text, str)
            or not isinstance(response_source, str)
            or not response_source.strip()
        ):
            raise ValueError("this task requires raw worker response text and source provenance")
        if (record.get("metadata") or {}).get("require_raw_response_provenance") is True and (
            not isinstance(raw_response_text, str)
            or not isinstance(response_source, str)
            or not response_source.strip()
        ):
            raise ValueError("this task requires raw worker response text and source provenance")

        try:
            normalized = json.loads(_canonical(response))
            if not isinstance(normalized, dict):
                raise ValueError("response must be a JSON object")
            if validator is not None and validator(normalized) is False:
                raise ValueError("response validator rejected the output")
        except (TypeError, ValueError) as error:
            record.pop("response", None)
            record.pop("response_sha256", None)
            record.pop("raw_response_text", None)
            record.pop("raw_response_sha256", None)
            record.pop("response_source", None)
            record["validation_error_type"] = type(error).__name__
            if record["repair_count"] < self.max_repairs:
                record["repair_count"] += 1
                record["status"] = "repair_required"
            else:
                record["status"] = "failed"
            _atomic_json(path, record)
            return record

        record["response"] = normalized
        record["response_sha256"] = _sha256_text(_canonical(normalized))
        if raw_response_text is not None or response_source is not None:
            if not isinstance(raw_response_text, str) or not isinstance(response_source, str) or not response_source.strip():
                raise ValueError("raw_response_text and response_source must be provided together")
            raw_response = json.loads(raw_response_text)
            raw_matches = raw_response == normalized
            if not raw_matches and (
                    (record.get("metadata") or {}).get("require_raw_response_provenance") is True
                    and isinstance(raw_response, dict)):
                provenance = {
                    "task_id": raw_response.get("task_id"),
                    "request_sha256": raw_response.get("request_sha256"),
                    "worker_context_id": raw_response.get("worker_context_id"),
                }
                expected = {
                    "task_id": task_id,
                    "request_sha256": record.get("request_sha256"),
                    "worker_context_id": assigned_worker,
                }
                model_response = {
                    key: value for key, value in raw_response.items()
                    if key not in {"task_id", "request_sha256", "worker_context_id"}
                }
                optional_bindings = {
                    "task_identity_sha256": record.get("task_identity_sha256"),
                    "case_id": (record.get("metadata") or {}).get("case_id"),
                    "dataset": (record.get("metadata") or {}).get("dataset"),
                    "horizon": (record.get("metadata") or {}).get("horizon"),
                    "revision_id": (record.get("metadata") or {}).get("memory_version"),
                }
                optional_fields_valid = True
                for key, expected_value in optional_bindings.items():
                    if key in model_response:
                        if expected_value is None or model_response[key] != expected_value:
                            optional_fields_valid = False
                            break
                        model_response.pop(key)
                raw_matches = provenance == expected and model_response == normalized
                raw_matches = raw_matches and optional_fields_valid
            if not raw_matches:
                raise ValueError("raw worker response does not match normalized response")
            record["raw_response_text"] = raw_response_text
            record["raw_response_sha256"] = _sha256_text(raw_response_text)
            record["response_source"] = response_source
        record["status"] = "completed"
        record.pop("validation_error_type", None)
        _atomic_json(path, record)
        return record

    def reject_response(self, task_id: str, *, error_type: str) -> dict:
        """Return a completed response to the one-repair queue after local validation."""
        if not isinstance(error_type, str) or not error_type.strip():
            raise ValueError("error_type must be a non-empty string")
        path = self._path(task_id)
        record = self._read(path)
        if record["status"] != "completed":
            raise RuntimeError("only a completed response can be rejected")
        record.pop("response", None)
        record.pop("response_sha256", None)
        record.pop("raw_response_text", None)
        record.pop("raw_response_sha256", None)
        record.pop("response_source", None)
        record["validation_error_type"] = error_type
        if record["repair_count"] < self.max_repairs:
            record["repair_count"] += 1
            record["status"] = "repair_required"
        else:
            record["status"] = "failed"
        _atomic_json(path, record)
        return record

    def mark_persisted(self, request_sha256: str) -> dict:
        if not isinstance(request_sha256, str) or not request_sha256.strip():
            raise ValueError("request_sha256 must be non-empty text")
        matches = [
            (path, self._read(path))
            for path in self.root.glob("*.json")
            if self._read(path).get("request_sha256") == request_sha256
        ]
        if not matches:
            raise KeyError("request identity is not present in this Codex task queue")
        if len(matches) != 1:
            raise ValueError("request identity maps to multiple Codex tasks")
        path, record = matches[0]
        if record["status"] not in {"completed", "persisted"}:
            raise RuntimeError("only a completed Codex response can be marked persisted")
        record["status"] = "persisted"
        _atomic_json(path, record)
        return record

    def mark_failed(self, task_id: str, *, error_type: str) -> dict:
        if not isinstance(error_type, str) or not error_type.strip():
            raise ValueError("error_type must be a non-empty string")
        path = self._path(task_id)
        record = self._read(path)
        if record["status"] != "dispatched":
            raise RuntimeError("only a dispatched task can be marked failed")
        record["status"] = "failed"
        record["worker_error_type"] = error_type
        _atomic_json(path, record)
        return record

    def requeue_dispatched(self, task_id: str, *, worker_error_type: str) -> dict:
        """Requeue a dispatch that terminated without producing a response."""
        if not isinstance(worker_error_type, str) or not worker_error_type.strip():
            raise ValueError("worker_error_type must be non-empty text")
        path = self._path(task_id)
        record = self._read(path)
        if record["status"] != "dispatched":
            raise RuntimeError("only a dispatched task can be requeued")
        record["status"] = "pending"
        record.setdefault("dispatch_failures", []).append(worker_error_type.strip())
        record.pop("worker_context_id", None)
        record.pop("context_isolation_mode", None)
        _atomic_json(path, record)
        return record

    def recover_dispatched(
        self,
        task_id: str,
        *,
        expected_worker_context_id: str,
        expected_task_identity_sha256: str,
        reason: str,
    ) -> dict:
        """Recover one caller-confirmed lost worker without changing its request.

        The caller must first check for a successful worker output. This method
        performs no liveness probes, timeout guesses, scans, or redispatches.
        Repeating a recovery is rejected once its dispatch is no longer current.
        As with other queue mutations, callers must serialize writes to one task.
        """
        for name, value in (
            ("expected_worker_context_id", expected_worker_context_id),
            ("expected_task_identity_sha256", expected_task_identity_sha256),
            ("reason", reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")

        record = self.get(task_id)
        if record["status"] != "dispatched":
            raise RuntimeError("only a dispatched task can be recovered")
        if (
            record.get("task_id_sha256") != self._task_key(task_id)
            or record.get("task_identity_sha256") != expected_task_identity_sha256
        ):
            raise ValueError("task identity does not match the expected recovery identity")
        if record.get("worker_context_id") != expected_worker_context_id:
            raise ValueError("worker_context_id does not match the expected lost worker")
        if "response" in record or "response_sha256" in record:
            raise RuntimeError("a task with a saved response cannot be recovered")

        record.setdefault("recovery_history", []).append({
            "task_id": task_id,
            "task_id_sha256": record["task_id_sha256"],
            "task_identity_sha256": record["task_identity_sha256"],
            "request_sha256": record.get("request_sha256"),
            "worker_context_id": record["worker_context_id"],
            "context_isolation_mode": record.get("context_isolation_mode"),
            "dispatch_count": record["dispatch_count"],
            "repair_count": record["repair_count"],
            "previous_status": record["status"],
            "status": "pending",
            "reason": reason.strip(),
            "recovered_utc": datetime.now(timezone.utc).isoformat(),
        })
        record["status"] = "pending"
        record.pop("worker_context_id", None)
        record.pop("context_isolation_mode", None)
        _atomic_json(self._path(task_id), record)
        return record


class CodexTaskPending(RuntimeError):
    """Raised when the HIC runner reaches a task awaiting Codex-worker output."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"Codex task is pending: {task_id}")


def wait_for_agent_response(queue, task_id, *, poll_seconds=2.0, sleep_fn=time.sleep):
    """Wait for a dispatched task response while preserving the caller process."""
    if not isinstance(poll_seconds, (int, float)) or isinstance(poll_seconds, bool):
        raise ValueError("poll_seconds must be a finite positive number")
    if not math.isfinite(float(poll_seconds)) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be a finite positive number")
    if not callable(sleep_fn):
        raise ValueError("sleep_fn must be callable")
    while True:
        record = queue.get(task_id)
        status = record.get("status")
        if status in {"completed", "persisted"}:
            return record
        if status == "failed":
            raise RuntimeError(f"Agent task failed: {task_id}")
        if status not in {"pending", "dispatched", "repair_required"}:
            raise ValueError(f"unsupported Agent task status while waiting: {status}")
        sleep_fn(float(poll_seconds))


def run_until_agent_ready(
    run_stage,
    queue,
    *,
    wait,
    on_pending=None,
    poll_seconds=2.0,
    sleep_fn=time.sleep,
):
    """Retry a stage after its Codex task response is persisted in the queue."""
    if not callable(run_stage):
        raise ValueError("run_stage must be callable")
    if type(wait) is not bool:
        raise ValueError("wait must be a boolean")
    if on_pending is not None and not callable(on_pending):
        raise ValueError("on_pending must be callable")

    while True:
        try:
            return run_stage()
        except CodexTaskPending as error:
            record = dict(queue.get(error.task_id))
            if record.get("task_id") not in (None, error.task_id):
                raise ValueError("Agent queue returned a different task identity")
            record["task_id"] = error.task_id
            if on_pending is not None:
                on_pending(record)
            if not wait:
                raise
            wait_for_agent_response(
                queue,
                error.task_id,
                poll_seconds=poll_seconds,
                sleep_fn=sleep_fn,
            )


class CodexTaskRequestCache:
    """RequestCache-compatible adapter that pauses instead of calling an API."""

    def __init__(
        self,
        queue: CodexTaskQueue,
        *,
        run_id: str,
        dataset: str,
        horizon: int,
        frequency: str,
        target: str,
        unit: str,
        prompt_version: str = "hic-codex-worker-v1",
        memory_version: str | None = None,
        require_raw_response_provenance: bool = False,
    ):
        if not isinstance(queue, CodexTaskQueue):
            raise TypeError("queue must be a CodexTaskQueue")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if memory_version is not None and (
            not isinstance(memory_version, str) or not memory_version.strip()
        ):
            raise ValueError("memory_version must be non-empty text when provided")
        if type(require_raw_response_provenance) is not bool:
            raise ValueError("require_raw_response_provenance must be a boolean")
        self.queue = queue
        self.run_id = run_id
        self.dataset = dataset
        self.horizon = horizon
        self.frequency = frequency
        self.target = target
        self.unit = unit
        self.prompt_version = prompt_version
        self.memory_version = memory_version
        self.require_raw_response_provenance = require_raw_response_provenance
        self.request_settings = {
            "provider": BACKEND,
            "model": "codex-task-worker",
            "run_id": run_id,
            "dataset": dataset,
            "horizon": horizon,
            "frequency": frequency,
            "target": target,
            "unit": unit,
            "prompt_version": prompt_version,
            "require_raw_response_provenance": require_raw_response_provenance,
        }
        if memory_version is not None:
            self.request_settings["memory_version"] = memory_version
        self.logical_calls = 0

    def _task_details(self, call_id: str) -> tuple[str, str]:
        if call_id.startswith(f"{self.run_id}:"):
            suffix = call_id[len(self.run_id) + 1 :]
        else:
            suffix = call_id
        parts = suffix.split(":")
        case_id = parts[0]
        stage_parts = parts[1:]
        horizon_token = f"h{self.horizon}"
        if stage_parts and stage_parts[0].lower() == horizon_token:
            stage_parts = stage_parts[1:]
        stage_token = ":".join(stage_parts).lower()
        if "summary" in stage_token:
            task_type = "summary"
        elif "baseline" in stage_token or "forecast" in stage_token:
            task_type = "baseline"
        elif "reflection" in stage_token or "memory" in stage_token:
            task_type = "fit_reflection"
        else:
            task_type = "correction"
        return case_id, task_type

    def invoke(
        self,
        call_id,
        prompt,
        *,
        call_json,
        validator=None,
        retry_unresolved=False,
    ):
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be non-empty text")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if not callable(call_json):
            raise ValueError("call_json must be callable")
        if validator is not None and not callable(validator):
            raise ValueError("validator must be callable")

        case_id, task_type = self._task_details(call_id)
        identity = {
            "request_settings": self.request_settings,
            "task_id": call_id,
            "prompt_sha256": _sha256_text(prompt),
        }
        request_sha256 = _sha256_text(_canonical(identity))
        existing = self.queue.get(call_id) if _path_is_file(self.queue._path(call_id)) else None
        if existing is None:
            existing = self.queue.enqueue(
                call_id,
                task_type=task_type,
                prompt=prompt,
                prompt_version=self.prompt_version,
                output_schema=f"{task_type}-json-v1",
                metadata={
                    "run_id": self.run_id,
                    "case_id": case_id,
                    "dataset": self.dataset,
                    "horizon": self.horizon,
                    "frequency": self.frequency,
                    "target": self.target,
                    "unit": self.unit,
                    "stage": ":".join(
                        part for part in call_id.split(":")[2:]
                        if part.lower() != f"h{self.horizon}"
                    ) or task_type,
                    **(
                        {
                            "memory_version": self.memory_version,
                            "context_isolation_required": True,
                        }
                        if self.memory_version is not None
                        else {}
                    ),
                    "require_raw_response_provenance": self.require_raw_response_provenance,
                },
                request_sha256=request_sha256,
            )
        elif existing.get("request_sha256") != request_sha256:
            raise ValueError("Codex task request identity changed for an existing call_id")

        if existing["status"] in {"completed", "persisted"}:
            try:
                CodexTaskQueue._validate(existing["response"], validator)
            except Exception as error:
                rejected = self.queue.reject_response(
                    call_id, error_type=type(error).__name__
                )
                if rejected["status"] in {"repair_required", "pending"}:
                    raise CodexTaskPending(call_id) from error
                raise ValueError("Codex worker response failed local validation") from error
            return {
                "response": existing["response"],
                "request_identity": {
                    "sha256": request_sha256,
                    "prompt_sha256": _sha256_text(prompt),
                    "request_settings": self.request_settings,
                    "backend": BACKEND,
                },
            }
        if existing["status"] == "failed":
            raise RuntimeError(f"Codex task failed: {call_id}")
        if existing["status"] == "dispatched" and retry_unresolved:
            raise RuntimeError("dispatched Codex task is unresolved; inspect the task worker first")
        raise CodexTaskPending(call_id)

    def mark_persisted(self, request_sha256):
        return self.queue.mark_persisted(request_sha256)
