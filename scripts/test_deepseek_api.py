"""Make one minimal live DeepSeek Flash request without displaying the API key."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from LLM.deepseek_client import DEFAULT_MODEL, call_chat_completion


def read_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    env_path = ROOT / "config" / "api.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "DEEPSEEK_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise SystemExit(
        "DEEPSEEK_API_KEY is missing. Add it to config/api.env; the script never prints it."
    )


def main():
    content = call_chat_completion(
        'Return exactly a JSON object with one field: {"status":"ok"}.',
        api_key=read_api_key(),
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=64,
        max_retries=2,
    )
    import json

    result = json.loads(content)
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise SystemExit("API replied, but the JSON smoke-test response was unexpected.")
    print(f"DeepSeek Flash API OK (model={DEFAULT_MODEL}; JSON response parsed).")


if __name__ == "__main__":
    main()
