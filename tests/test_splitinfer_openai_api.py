import unittest
from tempfile import NamedTemporaryFile

from fastapi.testclient import TestClient

from SplitInfer.api_server import build_arg_parser, create_app
from SplitInfer.openai_api import (
    OpenAIRequestError,
    build_chat_response,
    build_stream_events,
    parse_chat_request,
    prepare_inference_input,
)
from SplitInfer.server_config import ServerSettings, load_server_settings, parse_model_path_overrides


class FakeService:
    def __init__(self):
        self.stream_called = False
        self.buffered_called = False

    def list_models(self):
        return ["Llama-3-8B-Instruct"]

    def run_chat_completion(self, model_name, messages, max_tokens, temperature):
        self.buffered_called = True
        return {
            "model": model_name,
            "content": "buffered response",
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }

    def stream_chat_completion(self, model_name, messages, max_tokens, temperature):
        self.stream_called = True
        for piece in ["hello", " ", "stream"]:
            yield piece


class OpenAIAPIHelpersTest(unittest.TestCase):
    def test_parse_chat_request_accepts_valid_payload(self):
        payload = {
            "model": "Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 16,
        }
        parsed = parse_chat_request(payload)
        self.assertEqual(parsed.model, "Llama-3-8B-Instruct")
        self.assertTrue(parsed.stream)
        self.assertEqual(parsed.max_tokens, 16)

    def test_prepare_inference_input_rejects_images_for_text_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                ],
            }
        ]
        with self.assertRaises(OpenAIRequestError):
            prepare_inference_input(messages, supports_images=False)

    def test_prepare_inference_input_flattens_messages(self):
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        prepared = prepare_inference_input(messages, supports_images=False)
        self.assertIn("System: You are concise.", prepared.prompt_text)
        self.assertTrue(prepared.prompt_text.endswith("Assistant:"))
        self.assertIsNone(prepared.image_url)

    def test_build_chat_response_contains_usage(self):
        response = build_chat_response(
            model="Llama-3-8B-Instruct",
            content="hello",
            prompt_tokens=10,
            completion_tokens=2,
            created=123,
            completion_id="chatcmpl-test",
        )
        self.assertEqual(response["id"], "chatcmpl-test")
        self.assertEqual(response["usage"]["total_tokens"], 12)

    def test_build_stream_events_finish_with_done(self):
        events = build_stream_events("Llama-3-8B-Instruct", "hello world", completion_id="chatcmpl-test")
        self.assertTrue(events[0].startswith("data: "))
        self.assertEqual(events[-1], "data: [DONE]\n\n")

    def test_parse_model_path_overrides(self):
        overrides = parse_model_path_overrides([
            "Llama-3-8B-Instruct=/models/llama",
            "Qwen2-VL-7B-Instruct=/models/qwen",
        ])
        self.assertEqual(overrides["Llama-3-8B-Instruct"], "/models/llama")
        self.assertEqual(overrides["Qwen2-VL-7B-Instruct"], "/models/qwen")

    def test_load_server_settings_applies_cli_overrides(self):
        settings = load_server_settings(
            cli_overrides={
                "host": "127.0.0.1",
                "port": 9000,
                "weights_root": "/srv/weights",
                "model_paths": {"Llama-3-8B-Instruct": "/srv/llama"},
            }
        )
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.weights_root, "/srv/weights")
        self.assertEqual(settings.model_paths["Llama-3-8B-Instruct"], "/srv/llama")

    def test_load_server_settings_keeps_config_host_port_without_cli_overrides(self):
        with NamedTemporaryFile("w+", suffix=".json") as handle:
            handle.write('{"host":"127.0.0.1","port":9100}')
            handle.flush()
            settings = load_server_settings(config_path=handle.name, cli_overrides={"host": None, "port": None})
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 9100)

    def test_build_arg_parser_keeps_host_port_unset_by_default(self):
        args = build_arg_parser().parse_args([])
        self.assertIsNone(args.host)
        self.assertIsNone(args.port)

    def test_streaming_endpoint_uses_stream_path(self):
        service = FakeService()
        client = TestClient(create_app(service=service, settings=ServerSettings()))
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "Llama-3-8B-Instruct",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(service.stream_called)
        self.assertFalse(service.buffered_called)
        self.assertIn('"content": "hello"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    def test_non_streaming_endpoint_uses_buffered_path(self):
        service = FakeService()
        client = TestClient(create_app(service=service, settings=ServerSettings()))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "Llama-3-8B-Instruct",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(service.buffered_called)
        self.assertFalse(service.stream_called)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "buffered response")


if __name__ == "__main__":
    unittest.main()
