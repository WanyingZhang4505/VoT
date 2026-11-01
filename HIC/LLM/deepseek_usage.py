"""Append-only, content-free DeepSeek usage ledger and cost estimator."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path


_SCHEMA_VERSION = 1
_PRICING_VERSION = "deepseek-flash-prices-verified-2026-10-18"
_FLASH_USD_PER_MILLION = {
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60},
    "peak": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
}
_FLASH_CNY_PEAK_PER_MILLION = {
    "cache_hit": Decimal("0.04"),
    "cache_miss": Decimal("2"),
    "output": Decimal("8"),
}
_CNY_BUDGET_SCHEMA_VERSION = 1
_CNY_BUDGET_PRICING_VERSION = "deepseek-flash-cny-peak-snapshot-2026-10-19"


class CNYBudgetBlocked(RuntimeError):
    """Raised when a DeepSeek request is rejected before HTTP submission."""


def _non_negative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalize_usage(usage):
    prompt = _non_negative_int(usage.get("prompt_tokens"), "prompt_tokens")
    completion = _non_negative_int(usage.get("completion_tokens"), "completion_tokens")
    total = usage.get("total_tokens", prompt + completion)
    total = _non_negative_int(total, "total_tokens")
    if total != prompt + completion:
        raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")

    details = usage.get("prompt_tokens_details")
    cached_from_details = details.get("cached_tokens") if isinstance(details, dict) else None
    hit_raw = usage.get("prompt_cache_hit_tokens", cached_from_details)
    miss_raw = usage.get("prompt_cache_miss_tokens")
    if hit_raw is None and miss_raw is None:
        hit, miss = 0, prompt
        cache_breakdown_available = False
    elif hit_raw is None:
        miss = _non_negative_int(miss_raw, "prompt_cache_miss_tokens")
        hit = prompt - miss
        if hit < 0:
            raise ValueError("prompt cache-miss tokens exceed prompt_tokens")
        cache_breakdown_available = True
    elif miss_raw is None:
        hit = _non_negative_int(hit_raw, "prompt_cache_hit_tokens")
        miss = prompt - hit
        if miss < 0:
            raise ValueError("prompt cache-hit tokens exceed prompt_tokens")
        cache_breakdown_available = True
    else:
        hit = _non_negative_int(hit_raw, "prompt_cache_hit_tokens")
        miss = _non_negative_int(miss_raw, "prompt_cache_miss_tokens")
        cache_breakdown_available = True

    if hit + miss != prompt:
        raise ValueError("cache-hit and cache-miss tokens must sum to prompt_tokens")

    completion_details = usage.get("completion_tokens_details")
    reasoning_raw = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict) else None
    )
    normalized = {
        "prompt_tokens": prompt,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_breakdown_available": cache_breakdown_available,
    }
    if reasoning_raw is not None:
        normalized["reasoning_tokens"] = _non_negative_int(reasoning_raw, "reasoning_tokens")
    return normalized


def _estimate_flash_cost(usage):
    if usage is None:
        return None
    if usage.get("model_family") != "deepseek-flash":
        return None
    estimates = {}
    for period, rates in _FLASH_USD_PER_MILLION.items():
        total = (
            usage["prompt_cache_hit_tokens"] * rates["cache_hit"]
            + usage["prompt_cache_miss_tokens"] * rates["cache_miss"]
            + usage["completion_tokens"] * rates["output"]
        ) / 1_000_000
        estimates[period] = round(total, 12)
    return estimates


def _model_family(model):
    if model in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}:
        return "deepseek-flash"
    return None


class DeepSeekUsageLedger:
    """Persist one sanitized event for each successful HTTP 200 response."""

    def __init__(self, path):
        self.path = Path(path)

    def record_http_200(self, response_metadata, *, observed_at=None):
        if not isinstance(response_metadata, dict):
            raise TypeError("response metadata must be a dictionary")
        now = observed_at or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        model = response_metadata.get("model")
        if not isinstance(model, str) or not model or len(model) > 120:
            model = "unknown"

        event = {
            "schema_version": _SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "observed_at_utc": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "http_status": 200,
            "model": model,
            "provider_created": response_metadata.get("created")
            if type(response_metadata.get("created")) is int else None,
            "usage": None,
            "estimated_cost_usd": None,
            "pricing_version": _PRICING_VERSION,
        }

        parse_error_type = response_metadata.get("_usage_parse_error_type")
        raw_usage = response_metadata.get("usage")
        if parse_error_type is not None:
            event["usage_error_type"] = str(parse_error_type)[:80]
        elif isinstance(raw_usage, dict):
            try:
                usage = _normalize_usage(raw_usage)
            except (TypeError, ValueError) as error:
                event["usage_error_type"] = type(error).__name__
            else:
                event["usage"] = usage
                family = _model_family(model)
                if family is not None:
                    event["estimated_cost_usd"] = _estimate_flash_cost(
                        {**usage, "model_family": family}
                    )
        else:
            event["usage_error_type"] = "UsageMissing"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return event


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    preserve_temporary = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if attempt == 4:
                    preserve_temporary = True
                    raise PermissionError(
                        f"atomic replace failed after 5 attempts; recovery file retained at {temporary}"
                    ) from error
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if not preserve_temporary:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_file_lock(path, *, timeout_seconds=120):
    """Serialize small ledger updates across processes on Windows and POSIX."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        deadline = time.monotonic() + timeout_seconds
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for the shared ledger lock") from None
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for the shared ledger lock") from None
                    time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _peak_cost_micros(usage):
    normalized = _normalize_usage(usage)
    rates = _FLASH_CNY_PEAK_PER_MILLION
    micros = (
        Decimal(normalized["prompt_cache_hit_tokens"]) * rates["cache_hit"]
        + Decimal(normalized["prompt_cache_miss_tokens"]) * rates["cache_miss"]
        + Decimal(normalized["completion_tokens"]) * rates["output"]
    )
    return int(micros.to_integral_value(rounding=ROUND_CEILING)), normalized


class DeepSeekCNYSpendBudget:
    """Fail-closed peak-rate CNY cap for each physical DeepSeek HTTP attempt.

    Prompt reservations use UTF-8 byte length plus a fixed chat-format allowance
    as an upper-bound token estimate, assume every input token is an uncached
    miss, and reserve the full requested output. No prompt text or API key is
    persisted. HTTP errors and interrupted requests retain their reservation.
    """

    def __init__(
        self, path, *, max_spend_cny, resume=False, allow_initialize_on_resume=False,
        allow_reconcile_unresolved=False,
        inherited_cost_cny="0", inherited_cost_source_sha256=None,
    ):
        if not isinstance(resume, bool):
            raise ValueError("resume must be a boolean")
        if not isinstance(allow_initialize_on_resume, bool):
            raise ValueError("allow_initialize_on_resume must be a boolean")
        if not isinstance(allow_reconcile_unresolved, bool):
            raise ValueError("allow_reconcile_unresolved must be a boolean")
        try:
            cap = Decimal(str(max_spend_cny))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("max_spend_cny must be a positive finite amount") from None
        if not cap.is_finite() or cap <= 0:
            raise ValueError("max_spend_cny must be a positive finite amount")
        cap_micros = int((cap * Decimal(1_000_000)).to_integral_value())
        if cap * Decimal(1_000_000) != cap_micros:
            raise ValueError("max_spend_cny may have at most six decimal places")
        try:
            inherited_cost = Decimal(str(inherited_cost_cny))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("inherited_cost_cny must be a non-negative finite amount") from None
        if not inherited_cost.is_finite() or inherited_cost < 0:
            raise ValueError("inherited_cost_cny must be a non-negative finite amount")
        inherited_micros = int((inherited_cost * Decimal(1_000_000)).to_integral_value())
        if inherited_cost * Decimal(1_000_000) != inherited_micros:
            raise ValueError("inherited_cost_cny may have at most six decimal places")
        if inherited_micros and (
            not isinstance(inherited_cost_source_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", inherited_cost_source_sha256)
        ):
            raise ValueError("a SHA-256 source binding is required for inherited cost")
        if inherited_micros > cap_micros:
            raise ValueError("inherited cost exceeds the cumulative CNY cap")

        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.cap_micros = cap_micros
        self.inherited_cost_micros = inherited_micros
        self.inherited_cost_source_sha256 = inherited_cost_source_sha256
        with exclusive_file_lock(self.lock_path):
            if self.path.exists():
                if not resume:
                    raise FileExistsError("CNY spend ledger already exists; resume the same phase")
                state = self._read_state_locked(allow_cap_increase=True)
                if (
                    state.get("inherited_cost_micros", 0) != self.inherited_cost_micros
                    or state.get("inherited_cost_source_sha256")
                    != self.inherited_cost_source_sha256
                ):
                    raise ValueError("CNY spend ledger inherited cost basis does not match")
                unresolved = []
                for attempt in state["attempts"]:
                    if attempt.get("status") in {"reserved", "held", "unknown"}:
                        attempt["status"] = "unknown"
                        unresolved.append(attempt.get("attempt_id"))
                if unresolved:
                    state["resume_blocked_unresolved_attempts"] = len(unresolved)
                    self.state = state
                    self._save()
                    if not allow_reconcile_unresolved:
                        raise RuntimeError(
                            "CNY spend ledger contains unresolved HTTP-attempt reservations; "
                            "reconcile them before resuming to avoid exceeding the spend cap"
                        )
                saved_cap_micros = state["max_spend_micros"]
                if saved_cap_micros < self.cap_micros:
                    state["max_spend_micros"] = self.cap_micros
                    state["max_spend_cny"] = str(cap)
                    self.state = state
                    self._save()
                self.state = state
            else:
                if resume and not allow_initialize_on_resume:
                    raise FileNotFoundError(
                        "authorized CNY spend ledger is missing; refusing to reset the cumulative budget"
                    )
                self.state = {
                    "schema_version": _CNY_BUDGET_SCHEMA_VERSION,
                    "pricing_version": _CNY_BUDGET_PRICING_VERSION,
                    "max_spend_micros": cap_micros,
                    "max_spend_cny": str(cap),
                    "inherited_cost_micros": inherited_micros,
                    "inherited_cost_cny": str(inherited_cost),
                    "inherited_cost_source_sha256": inherited_cost_source_sha256,
                    "reservation_policy": (
                        "peak tariff; all prompt tokens treated as uncached; UTF-8 bytes + 256 "
                        "token allowance; full requested output reserved per HTTP attempt"
                    ),
                    "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "attempts": [],
                }
                self._save()

    def _read_state_locked(self, *, allow_cap_increase=False):
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            raise RuntimeError("CNY spend ledger is unreadable") from None
        saved_cap_micros = state.get("max_spend_micros") if isinstance(state, dict) else None
        cap_matches = saved_cap_micros == self.cap_micros
        allowed_increase = (
            allow_cap_increase
            and type(saved_cap_micros) is int
            and saved_cap_micros < self.cap_micros
        )
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != _CNY_BUDGET_SCHEMA_VERSION
            or state.get("pricing_version") != _CNY_BUDGET_PRICING_VERSION
            or not (cap_matches or allowed_increase)
            or not isinstance(state.get("attempts"), list)
        ):
            raise ValueError("CNY spend ledger does not match the authorized budget policy")
        if any(not isinstance(attempt, dict) for attempt in state["attempts"]):
            raise ValueError("CNY spend ledger contains an invalid attempt record")
        return state

    def _refresh_locked(self):
        self.state = self._read_state_locked()

    def _save(self):
        _atomic_write_json(self.path, self.state)

    def _totals(self):
        spent = self.inherited_cost_micros
        reserved = 0
        for attempt in self.state["attempts"]:
            status = attempt.get("status")
            if status in {"reconciled", "over_reserved"}:
                spent += attempt.get("actual_cost_micros", 0)
            elif status == "conservatively_charged":
                spent += attempt.get("conservative_cost_micros", 0)
            elif status in {"reserved", "held", "unknown"}:
                reserved += attempt.get("reserved_cost_micros", 0)
        return spent, reserved

    def reserve_request(self, *, prompt, model, max_output_tokens):
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string for CNY spend reservation")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if _model_family(model) != "deepseek-flash":
            raise ValueError("CNY spend guard only supports the verified DeepSeek Flash price")
        input_token_upper_bound = len(prompt.encode("utf-8")) + 256
        rates = _FLASH_CNY_PEAK_PER_MILLION
        reservation = int((
            Decimal(input_token_upper_bound) * rates["cache_miss"]
            + Decimal(max_output_tokens) * rates["output"]
        ).to_integral_value(rounding=ROUND_CEILING))
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            if any(item.get("status") == "unknown" for item in self.state["attempts"]):
                raise CNYBudgetBlocked(
                    "CNY spend is blocked by an unresolved prior HTTP attempt; request was not sent"
                )
            spent, reserved = self._totals()
            if spent + reserved + reservation > self.cap_micros:
                raise CNYBudgetBlocked(
                    "CNY spend cap would be exceeded by the next HTTP attempt; "
                    "request was not sent"
                )
            attempt_id = uuid.uuid4().hex
            self.state["attempts"].append({
                "attempt_id": attempt_id,
                "status": "reserved",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "model": model,
                "input_token_upper_bound": input_token_upper_bound,
                "output_token_upper_bound": max_output_tokens,
                "reserved_cost_micros": reservation,
                "reserved_cost_cny": reservation / 1_000_000,
            })
            self._save()
        return attempt_id

    def cancel_unsubmitted(self, attempt_id):
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            attempt = self._find_attempt(attempt_id)
            if attempt.get("status") != "reserved":
                raise RuntimeError("only a not-yet-submitted CNY reservation can be cancelled")
            attempt["status"] = "cancelled_before_http"
            attempt["reserved_cost_micros"] = 0
            attempt["reserved_cost_cny"] = 0.0
            self._save()

    def reconcile_dns_failure_before_connection(self, attempt_id):
        """Release a reservation only when DNS failed before any HTTP connection."""
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            attempt = self._find_attempt(attempt_id)
            if attempt.get("status") != "unknown" or attempt.get("error_type") != "gaierror":
                raise RuntimeError(
                    "only an unknown pre-connect DNS failure (gaierror) can be reconciled"
                )
            attempt["status"] = "cancelled_before_http"
            attempt["reserved_cost_micros"] = 0
            attempt["reserved_cost_cny"] = 0.0
            attempt["reconciliation_reason"] = "DNS lookup failed before HTTP connection"
            attempt["reconciled_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )
            unresolved = sum(
                item.get("status") in {"reserved", "held", "unknown"}
                for item in self.state["attempts"]
            )
            self.state["resume_blocked_unresolved_attempts"] = unresolved
            self._save()

    def conservatively_charge_unknown_reservations(self):
        """Keep unknown attempts within the cap by charging their full reservation."""
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            settled = 0
            settled_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for attempt in self.state["attempts"]:
                if attempt.get("status") != "unknown":
                    continue
                reserved_micros = attempt.get("reserved_cost_micros")
                if type(reserved_micros) is not int or reserved_micros < 0:
                    raise RuntimeError(
                        "cannot conservatively settle an unknown attempt with an invalid reservation"
                    )
                attempt["status"] = "conservatively_charged"
                attempt["conservative_cost_micros"] = reserved_micros
                attempt["conservative_cost_cny"] = reserved_micros / 1_000_000
                attempt["conservative_settlement_reason"] = (
                    "Provider usage is unavailable; the full pre-request reservation "
                    "is retained as a conservative upper-bound charge."
                )
                attempt["conservatively_settled_at_utc"] = settled_at
                settled += 1

            unresolved = sum(
                item.get("status") in {"reserved", "held", "unknown"}
                for item in self.state["attempts"]
            )
            self.state["resume_blocked_unresolved_attempts"] = unresolved
            spent, reserved = self._totals()
            if spent + reserved > self.cap_micros:
                raise RuntimeError(
                    "conservative settlement would exceed the remaining CNY spend cap"
                )
            self._save()
            return settled

    def record_http_error(self, attempt_id, status_code):
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            attempt = self._find_attempt(attempt_id)
            if attempt.get("status") != "reserved":
                raise RuntimeError("HTTP error does not match an active CNY reservation")
            attempt["status"] = "held"
            attempt["http_status"] = status_code if type(status_code) is int else None
            self._save()

    def record_transport_error(self, attempt_id, error_type):
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            attempt = self._find_attempt(attempt_id)
            if attempt.get("status") != "reserved":
                raise RuntimeError("transport error does not match an active CNY reservation")
            attempt["status"] = "unknown"
            attempt["error_type"] = str(error_type)[:80]
            self._save()

    def reconcile_http_200(self, attempt_id, response_metadata):
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            attempt = self._find_attempt(attempt_id)
            if attempt.get("status") != "reserved":
                raise RuntimeError("HTTP 200 does not match an active CNY reservation")
            model = response_metadata.get("model") if isinstance(response_metadata, dict) else None
            usage = response_metadata.get("usage") if isinstance(response_metadata, dict) else None
            if _model_family(model) != "deepseek-flash" or not isinstance(usage, dict):
                attempt["status"] = "unknown"
                attempt["usage_missing_or_unsupported"] = True
                self._save()
                raise RuntimeError("successful DeepSeek response has unknown billable usage")
            actual_micros, normalized = _peak_cost_micros(usage)
            attempt["normalized_usage"] = normalized
            attempt["actual_cost_micros"] = actual_micros
            attempt["actual_cost_cny_peak_estimate"] = actual_micros / 1_000_000
            attempt["status"] = "reconciled" if actual_micros <= attempt["reserved_cost_micros"] else "over_reserved"
            self._save()
            spent, reserved = self._totals()
            if actual_micros > attempt["reserved_cost_micros"]:
                raise RuntimeError("actual usage exceeded its conservative CNY reservation")
            if spent + reserved > self.cap_micros:
                raise RuntimeError("reconciled usage exceeds the remaining CNY spend cap")
            return normalized

    def _find_attempt(self, attempt_id):
        for attempt in reversed(self.state["attempts"]):
            if attempt.get("attempt_id") == attempt_id:
                return attempt
        raise KeyError("CNY spend reservation was not found")

    def summary(self):
        with exclusive_file_lock(self.lock_path):
            self._refresh_locked()
            spent, reserved = self._totals()
            conservatively_charged = sum(
                item.get("conservative_cost_micros", 0)
                for item in self.state["attempts"]
                if item.get("status") == "conservatively_charged"
            )
            return {
                "pricing_version": _CNY_BUDGET_PRICING_VERSION,
                "cap_cny": self.cap_micros / 1_000_000,
                "inherited_cost_cny": self.inherited_cost_micros / 1_000_000,
                "inherited_cost_source_sha256": self.inherited_cost_source_sha256,
                "peak_rate_spend_estimate_cny": spent / 1_000_000,
                "conservatively_charged_upper_bound_cny": conservatively_charged / 1_000_000,
                "held_or_unresolved_reserve_cny": reserved / 1_000_000,
                "remaining_cny": (self.cap_micros - spent - reserved) / 1_000_000,
                "budget_records": len(self.state["attempts"]),
                "usage_unknown": any(
                    item.get("status") == "unknown" for item in self.state["attempts"]
                ),
            }


def summarize_usage_ledger(path):
    """Aggregate token totals and off-peak/peak scenario cost estimates."""
    source = Path(path)
    totals = {
        "http_200_events": 0,
        "usage_known_events": 0,
        "usage_missing_events": 0,
        "usage_invalid_events": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_events": 0,
        "estimated_cost_usd": {"all_off_peak": 0.0, "all_peak": 0.0},
        "by_model": {},
        "cost_note": (
            "Scenario estimates use the recorded DeepSeek Flash tariff snapshot. "
            "The actual bill depends on request time, Chinese public holidays, "
            "model pricing changes, and the provider dashboard. When cache-token "
            "breakdown is absent, input tokens are conservatively treated as misses."
        ),
    }

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"usage ledger contains invalid JSON on line {line_number}") from None
            if event.get("schema_version") != _SCHEMA_VERSION or event.get("http_status") != 200:
                raise ValueError(f"usage ledger has an unsupported event on line {line_number}")
            totals["http_200_events"] += 1
            model = event.get("model", "unknown")
            bucket = totals["by_model"].setdefault(model, {
                "http_200_events": 0,
                "usage_known_events": 0,
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "uncached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": {"all_off_peak": 0.0, "all_peak": 0.0},
            })
            bucket["http_200_events"] += 1
            usage = event.get("usage")
            if not isinstance(usage, dict):
                if event.get("usage_error_type") == "UsageMissing":
                    totals["usage_missing_events"] += 1
                else:
                    totals["usage_invalid_events"] += 1
                continue

            totals["usage_known_events"] += 1
            bucket["usage_known_events"] += 1
            for field, key in (
                ("prompt_tokens", "prompt_tokens"),
                ("prompt_cache_hit_tokens", "cached_prompt_tokens"),
                ("prompt_cache_miss_tokens", "uncached_prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                count = usage[field]
                totals[key] += count
                bucket[key] += count

            cost = event.get("estimated_cost_usd")
            if isinstance(cost, dict):
                totals["cost_events"] += 1
                for period, key in (("off_peak", "all_off_peak"), ("peak", "all_peak")):
                    value = float(cost[period])
                    totals["estimated_cost_usd"][key] += value
                    bucket["estimated_cost_usd"][key] += value

    totals["estimated_cost_usd"] = {
        key: round(value, 12) for key, value in totals["estimated_cost_usd"].items()
    }
    for bucket in totals["by_model"].values():
        bucket["estimated_cost_usd"] = {
            key: round(value, 12)
            for key, value in bucket["estimated_cost_usd"].items()
        }
    return totals


def main():
    parser = argparse.ArgumentParser(description="Summarize a local DeepSeek usage JSONL ledger.")
    parser.add_argument("--ledger", required=True, help="Path to a usage JSONL file")
    args = parser.parse_args()
    print(json.dumps(summarize_usage_ledger(args.ledger), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
