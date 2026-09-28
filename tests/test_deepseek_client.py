import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from LLM.deepseek_client import call_chat_completion


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class DeepSeekClientTests(unittest.TestCase):
    def test_sends_json_mode_request_and_returns_assistant_content(self):
        response = FakeResponse(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

        with patch("LLM.deepseek_client.urlopen", return_value=response) as open_url:
            result = call_chat_completion(
                "Return JSON", api_key="test-key", sleep_fn=lambda _seconds: None
            )

        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["messages"][0]["content"], "Return JSON")
        self.assertEqual(result, '{"ok": true}')

    def test_missing_key_fails_before_network_call(self):
        with patch("LLM.deepseek_client.urlopen") as open_url:
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                call_chat_completion("Return JSON", api_key="")
        open_url.assert_not_called()

    def test_retries_transient_http_status(self):
        overloaded = HTTPError(
            "https://api.deepseek.com/chat/completions", 503, "busy", {}, BytesIO(b"")
        )
        response = FakeResponse(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

        with patch(
            "LLM.deepseek_client.urlopen", side_effect=[overloaded, response]
        ) as open_url, patch("LLM.deepseek_client.time.sleep") as sleep:
            result = call_chat_completion(
                "Return JSON", api_key="test-key", sleep_fn=sleep
            )

        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(result, '{"ok": true}')

    def test_retries_once_with_clear_json_instruction_after_empty_content(self):
        empty_response = FakeResponse(
            {"choices": [{"message": {"content": ""}}]}
        )
        valid_response = FakeResponse(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

        with patch(
            "LLM.deepseek_client.urlopen",
            side_effect=[empty_response, valid_response],
        ) as open_url, patch("LLM.deepseek_client.time.sleep") as sleep:
            result = call_chat_completion(
                "Return JSON", api_key="test-key", sleep_fn=sleep
            )

        self.assertEqual(open_url.call_count, 2)
        retry_body = json.loads(open_url.call_args_list[1].args[0].data)
        self.assertIn("non-empty valid JSON object", retry_body["messages"][0]["content"])
        sleep.assert_called_once_with(1)
        self.assertEqual(result, '{"ok": true}')

    def test_falls_back_from_json_mode_after_repeated_empty_content(self):
        empty_response = FakeResponse(
            {"choices": [{"message": {"content": ""}}]}
        )
        valid_response = FakeResponse(
            {"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

        with patch(
            "LLM.deepseek_client.urlopen",
            side_effect=[empty_response, empty_response, valid_response],
        ) as open_url, patch("LLM.deepseek_client.time.sleep") as sleep:
            result = call_chat_completion(
                "Return JSON", api_key="test-key", sleep_fn=sleep
            )

        self.assertEqual(open_url.call_count, 3)
        first_body = json.loads(open_url.call_args_list[0].args[0].data)
        retry_body = json.loads(open_url.call_args_list[1].args[0].data)
        fallback_body = json.loads(open_url.call_args_list[2].args[0].data)
        self.assertEqual(first_body["response_format"], {"type": "json_object"})
        self.assertEqual(retry_body["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", fallback_body)
        self.assertIn("valid JSON", fallback_body["messages"][0]["content"])
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(result, '{"ok": true}')

    def test_empty_content_error_reports_safe_finish_diagnostics(self):
        empty_response = FakeResponse({
            "choices": [{
                "finish_reason": "insufficient_system_resource",
                "message": {"content": "", "reasoning_content": "hidden reasoning"},
            }]
        })
        with patch("LLM.deepseek_client.urlopen", return_value=empty_response):
            with self.assertRaisesRegex(
                RuntimeError,
                "finish_reason=insufficient_system_resource.*reasoning_content_present=True",
            ):
                call_chat_completion(
                    "Return JSON", api_key="test-key", max_retries=1,
                    sleep_fn=lambda _seconds: None,
                )


if __name__ == "__main__":
    unittest.main()
