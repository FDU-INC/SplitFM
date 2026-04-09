# SplitInfer API Server Guide

## Overview
`SplitInfer` now includes an inference-only HTTP server that exposes an OpenAI-compatible API. The new server is additive: existing scripts such as `python SplitInfer/infer.py ...` keep their current behavior.

## Supported Endpoints
- `GET /v1/models`
- `POST /v1/chat/completions`

## Supported Models
- `Llama-3-8B-Instruct`
- `DeepSeek-R1-Distill-Llama-8B`
- `Qwen2-VL-7B-Instruct`

## Install
Create and activate your Python environment first, then install the server dependencies:

```bash
pip install -r SplitInfer/requirements-api.txt
```

Install the model runtime dependencies already required by SplitInfer, and make the model weights available through one of the configuration methods described below.
If you plan to serve `Qwen2-VL-7B-Instruct`, also install the external package that provides `qwen_vl_utils`.

## Model Weights Configuration
The server no longer assumes a single hard-coded model weights location for deployment. You can configure weights paths in three ways:

1. CLI parameters
2. Environment variables
3. A JSON config file

Priority is: CLI overrides > environment variables / config file > repository default path.

If no override is provided, the server falls back to the repository default path `SplitInfer/weights/<model-name>/`.

### CLI examples

```bash
python SplitInfer/api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --weights-root /srv/splitinfer-weights
```

```bash
python SplitInfer/api_server.py \
  --model-path Llama-3-8B-Instruct=/srv/models/llama3 \
  --model-path DeepSeek-R1-Distill-Llama-8B=/srv/models/deepseek
```

### Environment variable examples

```bash
export SPLITINFER_WEIGHTS_ROOT=/srv/splitinfer-weights
export SPLITINFER_MODEL_PATHS_JSON='{"Qwen2-VL-7B-Instruct":"/srv/models/qwen2vl"}'
uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

### Config file example
See [api_server.example.json](api_server.example.json).

```bash
python SplitInfer/api_server.py --config SplitInfer/api_server.example.json
```

## Start the Server
Using `uvicorn`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

Using the Python entrypoint:

```bash
python SplitInfer/api_server.py --host 0.0.0.0 --port 8000 --gpu 0,1,2
```

## Optional API Key
If you want Bearer authentication, set a static key before startup:

```bash
export SPLITINFER_API_KEY=your-secret-key
uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

Clients must then send:

```text
Authorization: Bearer your-secret-key
```

## Non-Streaming Chat Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain split inference in one paragraph."}
    ],
    "max_tokens": 64
  }'
```

## Streaming Chat Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "Give me three concise deployment tips."}
    ],
    "stream": true,
    "max_tokens": 64
  }'
```

## Multimodal Qwen Example
For `Qwen2-VL-7B-Instruct`, send OpenAI-style `text` and `image_url` content parts:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {
            "type": "image_url",
            "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
          }
        ]
      }
    ],
    "max_tokens": 64
  }'
```

## Notes and Limitations
- This server implements a focused OpenAI-compatible subset, not the full OpenAI API.
- Only one image is supported per request.
- `system`, `user`, and `assistant` roles are supported.
- `temperature` is accepted, but the current model adapters still use greedy decoding.
- Existing CLI inference and all training flows remain unchanged.
- The current repository's Qwen2-VL path contains pre-existing runtime issues. This change does not modify those issues. If Qwen2-VL requests fail, evaluate the existing adapter implementation separately.

## Known Repository Issues Not Changed By This Feature
- File: `SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`, line 22.
  Problem: `configuration` is referenced, but the method initializes `self.configuration`. This can raise `NameError`.
  Suggested change: replace `configuration._attn_implementation = "flash_attention_2"` with `self.configuration._attn_implementation = "flash_attention_2"`.
- File: `SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`, line 33.
  Problem: `lm_head` is referenced, but the field stored on the instance is `self.lm_head`. This can raise `NameError`.
  Suggested change: replace `self.lm_head = lm_head.half().cuda(1)` with `self.lm_head = self.lm_head.half().cuda(1)`.
- File: `SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`, lines 59-60.
  Problem: `hidden_states` and `position_ids` stay on `cuda:0`, but `self.model_server` and `self.lm_head` are placed on `cuda:1`. This can cause cross-device runtime errors.
  Suggested change: move both tensors to the same device as `self.model_server`, for example `cuda(1)`.

## Troubleshooting
- `401 Unauthorized`: check `SPLITINFER_API_KEY` and the `Authorization` header.
- `404 model_not_found`: the `model` value must match one of `/v1/models`.
- `400 image_not_supported`: you sent an image to a text-only model.
- Weight loading errors usually mean the model files are missing under `SplitInfer/weights/`.
- Qwen2-VL failures may come from existing repository issues rather than the HTTP server layer.
