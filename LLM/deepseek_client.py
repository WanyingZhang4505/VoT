"""Small standard-library client for DeepSeek's OpenAI-compatible chat API."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
RETRY_STATUS = {429, 500, 502, 503, 504}


def call_chat_completion(
    prompt,
    *,
    api_key,
    model=DEFAULT_MODEL,
    temperature=0.3,
    max_tokens=2048,
    max_retries=6,
    timeout=120,
    sleep_fn=time.sleep,
):
    """Send a JSON-mode chat request and return the assistant's text content."""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set; configure it in config/api.env.")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1.")

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
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt + 1 < max_retries:
                sleep_fn(min(2**attempt, 32))
                continue
            raise RuntimeError(
                f"DeepSeek API returned HTTP {exc.code}; check the API key, model, "
                "quota, and request configuration."
            ) from None
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach the DeepSeek API ({exc.reason})."
            ) from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("DeepSeek API returned a non-JSON response.") from None

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
