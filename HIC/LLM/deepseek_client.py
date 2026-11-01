"""Small standard-library client for DeepSeek's OpenAI-compatible chat API."""

import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _redact_provider_value(value):
    value = " ".join(value.split())[:240]
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)


def _safe_http_error_diagnostics(error):
    """Return only allowlisted, redacted provider diagnostics."""
    diagnostics = {}
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("error", payload)
        if isinstance(detail, dict):
            for field in ("code", "type", "message", "msg"):
                value = detail.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                diagnostics[field] = _redact_provider_value(value)

    headers = getattr(error, "headers", None)
    if headers is not None:
        for request_id in (headers.get("X-Request-Id"), headers.get("Request-Id")):
            if (
                isinstance(request_id, str)
                and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id)
                and not request_id.lower().startswith("sk-")
            ):
                diagnostics["request_id"] = request_id
                break

    return diagnostics


def _safe_http_error_context(error):
    """Format safe diagnostics for the human-readable exception message."""
    return "; ".join(
        f"{field}={value}"
        for field, value in _safe_http_error_diagnostics(error).items()
    )


class DeepSeekHTTPError(RuntimeError):
    """HTTP failure with structured, allowlisted provider diagnostics."""

    def __init__(self, status_code, provider_diagnostics):
        self.status_code = status_code
        self.provider_diagnostics = dict(provider_diagnostics)
        detail = "; ".join(
            f"{field}={value}"
            for field, value in self.provider_diagnostics.items()
        )
        message = f"DeepSeek API returned HTTP {status_code}."
        if detail:
            message += f" Provider diagnostics: {detail}."
        super().__init__(message)


class HttpAttemptBudget:
    """Shared, explicit ceiling for physical HTTP attempts across logical calls."""

    def __init__(self, *, max_attempts, attempts_used=0, on_consume=None):
        if not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("max_attempts must be a non-negative integer")
        if not isinstance(attempts_used, int) or not 0 <= attempts_used <= max_attempts:
            raise ValueError("attempts_used must be between zero and max_attempts")
        if on_consume is not None and not callable(on_consume):
            raise TypeError("on_consume must be callable")
        self.max_attempts = max_attempts
        self.attempts_used = attempts_used
        self.on_consume = on_consume

    def consume(self):
        if self.attempts_used >= self.max_attempts:
            raise RuntimeError(
                f"HTTP attempt budget ({self.max_attempts}) exhausted before request"
            )
        self.attempts_used += 1
        if self.on_consume is not None:
            self.on_consume(self.attempts_used)


def call_chat_completion(
    prompt,
    *,
    api_key,
    model=DEFAULT_MODEL,
    temperature=0.3,
    max_tokens=2048,
    max_retries=6,
    attempt_budget=None,
    spend_budget=None,
    usage_callback=None,
    timeout=120,
    sleep_fn=time.sleep,
):
    """Send a JSON-mode chat request and return the assistant's text content."""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set; configure it in config/api.env.")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")
    if usage_callback is not None and not callable(usage_callback):
        raise ValueError("usage_callback must be callable")
    if spend_budget is not None and not all(
        callable(getattr(spend_budget, name, None))
        for name in (
            "reserve_request", "cancel_unsubmitted", "record_http_error",
            "record_transport_error", "reconcile_http_200",
        )
    ):
        raise ValueError("spend_budget does not implement the required CNY spend interface")

    current_prompt = prompt
    empty_response_retries = 0
    json_mode = True
    empty_finish_reason = "unknown"
    reasoning_content_present = False
    for attempt in range(max_retries):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": current_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = Request(
            API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        reservation_id = None
        if spend_budget is not None:
            reservation_id = spend_budget.reserve_request(
                prompt=current_prompt,
                model=model,
                max_output_tokens=max_tokens,
            )
        try:
            if attempt_budget is not None:
                attempt_budget.consume()
        except Exception:
            if reservation_id is not None:
                spend_budget.cancel_unsubmitted(reservation_id)
            raise
        try:
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                raw_body = response.read()
        except HTTPError as exc:
            if reservation_id is not None:
                spend_budget.record_http_error(reservation_id, exc.code)
            if exc.code in RETRY_STATUS and attempt + 1 < max_retries:
                sleep_fn(min(2**attempt, 32))
                continue
            raise DeepSeekHTTPError(
                exc.code, _safe_http_error_diagnostics(exc)
            ) from None
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            reason_type = type(reason).__name__
            if reservation_id is not None:
                spend_budget.record_transport_error(reservation_id, reason_type)
            raise RuntimeError(
                f"Could not reach the DeepSeek API ({reason_type})."
            ) from None

        if status != 200 and reservation_id is not None:
            spend_budget.record_http_error(reservation_id, status)
        if status != 200:
            raise RuntimeError(f"DeepSeek API returned unexpected HTTP status {status}.")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if status == 200 and usage_callback is not None:
                response_metadata = {
                    "model": model,
                    "created": None,
                    "usage": None,
                    "_usage_parse_error_type": type(error).__name__,
                }
                usage_callback(response_metadata)
                if reservation_id is not None:
                    spend_budget.reconcile_http_200(reservation_id, response_metadata)
            raise RuntimeError("DeepSeek API returned a non-JSON response.") from None

        if status == 200:
            response_metadata = {
                "model": payload.get("model", model) if isinstance(payload, dict) else model,
                "created": payload.get("created") if isinstance(payload, dict) else None,
                "usage": payload.get("usage") if isinstance(payload, dict) else None,
            }
            if usage_callback is not None:
                usage_callback(response_metadata)
            if reservation_id is not None:
                spend_budget.reconcile_http_200(reservation_id, response_metadata)

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "DeepSeek API response did not contain assistant content."
            ) from None
        if isinstance(content, str) and content.strip():
            return content
        empty_finish_reason = choice.get("finish_reason") or "unknown"
        reasoning_content_present = bool(message.get("reasoning_content"))
        if attempt + 1 < max_retries and empty_response_retries < 2:
            empty_response_retries += 1
            if empty_response_retries == 1:
                current_prompt = (
                    f"{prompt}\n\nYour previous response was empty. Return a non-empty "
                    "valid JSON object only, with no prose or Markdown fences."
                )
            else:
                json_mode = False
                current_prompt = (
                    f"{prompt}\n\nReturn a non-empty valid JSON object only. "
                    "Do not include prose or Markdown fences."
                )
            sleep_fn(min(2**attempt, 32))
            continue
        raise RuntimeError(
            "DeepSeek API returned empty assistant content "
            f"(finish_reason={empty_finish_reason}, "
            f"reasoning_content_present={reasoning_content_present})."
        )
