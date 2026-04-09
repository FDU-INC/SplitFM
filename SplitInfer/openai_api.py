import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional


ALLOWED_CHAT_REQUEST_KEYS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "temperature",
    "user",
}


ROLE_PREFIXES = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


class OpenAIRequestError(ValueError):
    def __init__(self, message: str, param: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


@dataclass
class ParsedMessage:
    role: str
    text: str
    image_urls: List[str]


@dataclass
class ParsedChatRequest:
    model: str
    messages: List[Dict[str, Any]]
    stream: bool
    max_tokens: Optional[int]
    temperature: Optional[float]


@dataclass
class PreparedInferenceInput:
    prompt_text: str
    image_url: Optional[str]


def parse_chat_request(payload: Dict[str, Any]) -> ParsedChatRequest:
    unknown_keys = set(payload.keys()) - ALLOWED_CHAT_REQUEST_KEYS
    if unknown_keys:
        key_list = ", ".join(sorted(unknown_keys))
        raise OpenAIRequestError(
            f"Unsupported request field(s): {key_list}.",
            param=sorted(unknown_keys)[0],
            code="unsupported_field",
        )

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise OpenAIRequestError("`model` must be a non-empty string.", param="model")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenAIRequestError("`messages` must be a non-empty array.", param="messages")

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise OpenAIRequestError("`stream` must be a boolean.", param="stream")

    max_tokens = payload.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise OpenAIRequestError("`max_tokens` must be a positive integer.", param="max_tokens")

    temperature = payload.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise OpenAIRequestError("`temperature` must be a number.", param="temperature")

    return ParsedChatRequest(
        model=model.strip(),
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
        temperature=float(temperature) if temperature is not None else None,
    )


def parse_message(message: Dict[str, Any]) -> ParsedMessage:
    if not isinstance(message, dict):
        raise OpenAIRequestError("Each message must be an object.", param="messages")

    role = message.get("role")
    if role not in ROLE_PREFIXES:
        raise OpenAIRequestError(
            "Supported roles are `system`, `user`, and `assistant`.",
            param="messages.role",
        )

    content = message.get("content")
    if isinstance(content, str):
        return ParsedMessage(role=role, text=content, image_urls=[])

    if not isinstance(content, list) or not content:
        raise OpenAIRequestError(
            "Message `content` must be a string or a non-empty array.",
            param="messages.content",
        )

    text_parts: List[str] = []
    image_urls: List[str] = []

    for part in content:
        if not isinstance(part, dict):
            raise OpenAIRequestError("Each content part must be an object.", param="messages.content")

        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise OpenAIRequestError("Text content parts must include a string `text`.", param="messages.content.text")
            text_parts.append(text)
            continue

        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str) or not image_url.strip():
                raise OpenAIRequestError(
                    "Image content parts must include a non-empty `image_url`.",
                    param="messages.content.image_url",
                )
            image_urls.append(image_url.strip())
            continue

        raise OpenAIRequestError(
            "Only `text` and `image_url` content parts are supported.",
            param="messages.content.type",
        )

    return ParsedMessage(role=role, text="\n".join(text_parts).strip(), image_urls=image_urls)


def prepare_inference_input(messages: List[Dict[str, Any]], supports_images: bool) -> PreparedInferenceInput:
    parsed_messages = [parse_message(message) for message in messages]

    image_urls: List[str] = []
    prompt_lines: List[str] = []
    for parsed in parsed_messages:
        if parsed.image_urls:
            if not supports_images:
                raise OpenAIRequestError(
                    "The selected model does not support image inputs.",
                    param="messages.content",
                    code="image_not_supported",
                )
            image_urls.extend(parsed.image_urls)

        prefix = ROLE_PREFIXES[parsed.role]
        if parsed.text:
            prompt_lines.append(f"{prefix}: {parsed.text}")
        else:
            prompt_lines.append(f"{prefix}:")

    if len(image_urls) > 1:
        raise OpenAIRequestError(
            "Only one image is supported per request.",
            param="messages.content.image_url",
            code="too_many_images",
        )

    prompt_lines.append("Assistant:")
    prompt_text = "\n\n".join(prompt_lines)
    image_url = image_urls[0] if image_urls else None
    return PreparedInferenceInput(prompt_text=prompt_text, image_url=image_url)


def estimate_token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())

    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def build_models_response(model_ids: Iterable[str]) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "splitinfer",
            }
            for model_id in model_ids
        ],
    }


def build_chat_response(
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: Optional[int] = None,
    completion_id: Optional[str] = None,
) -> Dict[str, Any]:
    created = created or int(time.time())
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def iter_stream_events(
    model: str,
    pieces: Iterable[str],
    completion_id: Optional[str] = None,
) -> Iterator[str]:
    created = int(time.time())
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex}"

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    for piece in pieces:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def build_stream_events(model: str, content: str, completion_id: Optional[str] = None) -> List[str]:
    return list(iter_stream_events(model, chunk_text(content), completion_id=completion_id))


def chunk_text(text: str, chunk_size: int = 32) -> Iterable[str]:
    if not text:
        return [""]
    return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]
