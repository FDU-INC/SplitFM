import argparse
import os
from typing import Any, Dict, Optional

try:
    from .openai_api import (
        OpenAIRequestError,
        build_chat_response,
        build_models_response,
        estimate_token_count,
        iter_stream_events,
        parse_chat_request,
        prepare_inference_input,
    )
    from .runtime import MODEL_REGISTRY, ModelManager
    from .server_config import ServerSettings, load_server_settings, parse_model_path_overrides
except ImportError:
    from openai_api import (
        OpenAIRequestError,
        build_chat_response,
        build_models_response,
        estimate_token_count,
        iter_stream_events,
        parse_chat_request,
        prepare_inference_input,
    )
    from runtime import MODEL_REGISTRY, ModelManager
    from server_config import ServerSettings, load_server_settings, parse_model_path_overrides

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


class SplitInferenceService:
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()

    def list_models(self):
        return self.model_manager.list_models()

    def run_chat_completion(
        self,
        model_name: str,
        messages,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        loaded, prepared, infer_kwargs = self._prepare_completion(model_name, messages, max_tokens, temperature)
        content = loaded.adapter.infer(prepared.prompt_text, **infer_kwargs)
        prompt_tokens = estimate_token_count(loaded.tokenizer, prepared.prompt_text)
        completion_tokens = estimate_token_count(loaded.tokenizer, content)
        return {
            "model": model_name,
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def stream_chat_completion(
        self,
        model_name: str,
        messages,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ):
        loaded, prepared, infer_kwargs = self._prepare_completion(model_name, messages, max_tokens, temperature)
        return loaded.adapter.stream_infer(prepared.prompt_text, **infer_kwargs)

    def _prepare_completion(
        self,
        model_name: str,
        messages,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ):
        if model_name not in MODEL_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Unknown model `{model_name}`.")

        loaded = self.model_manager.get(model_name)
        prepared = prepare_inference_input(messages, supports_images=loaded.supports_images)
        infer_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "log_tokens": False,
        }
        if prepared.image_url is not None:
            infer_kwargs["image_path"] = prepared.image_url
        return loaded, prepared, infer_kwargs


def _openai_error_response(status_code: int, message: str, param: Optional[str] = None, code: Optional[str] = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": param,
                "code": code,
            }
        },
    )

def create_app(
    service: Optional[SplitInferenceService] = None,
    settings: Optional[ServerSettings] = None,
) -> FastAPI:
    settings = settings or load_server_settings()
    api_key = settings.api_key
    service = service or SplitInferenceService(
        model_manager=ModelManager(
            weights_root=settings.weights_root,
            model_paths=settings.model_paths,
        )
    )
    app = FastAPI(title="SplitInfer OpenAI-Compatible API", version="1.0.0")

    def require_api_key(authorization: Optional[str] = Header(default=None)):
        if not api_key:
            return

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

        token = authorization[len("Bearer "):].strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key.")

    @app.exception_handler(OpenAIRequestError)
    async def handle_request_error(_: Request, exc: OpenAIRequestError):
        return _openai_error_response(400, exc.message, param=exc.param, code=exc.code)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_: Request, exc: HTTPException):
        if exc.status_code == 401:
            return _openai_error_response(401, str(exc.detail), code="unauthorized")
        if exc.status_code == 404:
            return _openai_error_response(404, str(exc.detail), param="model", code="model_not_found")
        return _openai_error_response(exc.status_code, str(exc.detail))

    @app.get("/v1/models")
    async def list_models(_: None = Depends(require_api_key)):
        return build_models_response(service.list_models())

    @app.post("/v1/chat/completions")
    async def create_chat_completion(payload: Dict[str, Any], _: None = Depends(require_api_key)):
        parsed = parse_chat_request(payload)

        if parsed.stream:
            token_stream = service.stream_chat_completion(
                model_name=parsed.model,
                messages=parsed.messages,
                max_tokens=parsed.max_tokens,
                temperature=parsed.temperature,
            )
            return StreamingResponse(
                iter_stream_events(parsed.model, token_stream),
                media_type="text/event-stream",
            )

        result = service.run_chat_completion(
            model_name=parsed.model,
            messages=parsed.messages,
            max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
        )

        return build_chat_response(
            model=parsed.model,
            content=result["content"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )

    return app


app = create_app()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="SplitInfer OpenAI-compatible API server")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES override for the server process")
    parser.add_argument("--api-key", type=str, default=None, help="Optional static API key")
    parser.add_argument("--weights-root", type=str, default=None, help="Base directory that contains per-model weights folders")
    parser.add_argument("--model-path", action="append", default=None, help="Per-model weights override in MODEL=PATH format; repeatable")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file for server settings")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    settings = load_server_settings(
        config_path=args.config,
        cli_overrides={
            "host": args.host,
            "port": args.port,
            "gpu": args.gpu,
            "api_key": args.api_key,
            "weights_root": args.weights_root,
            "model_paths": parse_model_path_overrides(args.model_path),
        },
    )

    if settings.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu
    if settings.api_key:
        os.environ["SPLITINFER_API_KEY"] = settings.api_key

    import uvicorn

    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
