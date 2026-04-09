# SplitInfer API 服务指南

## 概述
`SplitInfer` 现在提供了一个仅用于推理的 HTTP 服务，接口兼容 OpenAI API 的常用格式。该功能是增量新增的，现有脚本例如 `python SplitInfer/infer.py ...` 的行为保持不变。

## 已支持的接口
- `GET /v1/models`
- `POST /v1/chat/completions`

## 已支持的模型
- `Llama-3-8B-Instruct`
- `DeepSeek-R1-Distill-Llama-8B`
- `Qwen2-VL-7B-Instruct`

## 安装
先准备好 Python 环境，然后安装服务端依赖：

```bash
pip install -r SplitInfer/requirements-api.txt
```

同时需要安装 SplitInfer 原本依赖的模型运行环境，并通过下文介绍的任一种配置方式提供模型权重路径。
如果你需要提供 `Qwen2-VL-7B-Instruct` 的多模态服务，还需要额外安装提供 `qwen_vl_utils` 的外部包。

## 模型权重配置
部署时，服务端不应依赖单一固定的模型权重目录。现在可以通过以下三种方式配置模型参数位置：

1. 命令行参数
2. 环境变量
3. JSON 配置文件

优先级为：命令行参数 > 环境变量 / 配置文件 > 仓库默认路径。

如果没有提供任何覆盖配置，服务端会回退到仓库默认路径 `SplitInfer/weights/<model-name>/`。

### 命令行示例

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

### 环境变量示例

```bash
export SPLITINFER_WEIGHTS_ROOT=/srv/splitinfer-weights
export SPLITINFER_MODEL_PATHS_JSON='{"Qwen2-VL-7B-Instruct":"/srv/models/qwen2vl"}'
uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

### 配置文件示例
参考 [api_server.example.json](api_server.example.json)。

```bash
python SplitInfer/api_server.py --config SplitInfer/api_server.example.json
```

## 启动服务
使用 `uvicorn`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

使用 Python 入口脚本：

```bash
python SplitInfer/api_server.py --host 0.0.0.0 --port 8000 --gpu 0,1,2
```

## 可选鉴权
如果你需要静态 Bearer Token 鉴权，可以在启动前设置：

```bash
export SPLITINFER_API_KEY=your-secret-key
uvicorn SplitInfer.api_server:app --host 0.0.0.0 --port 8000
```

客户端随后需要携带：

```text
Authorization: Bearer your-secret-key
```

## 非流式文本请求示例

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

## 流式请求示例

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

## Qwen 多模态示例
对于 `Qwen2-VL-7B-Instruct`，请求体使用 OpenAI 风格的 `text` 和 `image_url` 内容片段：

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

## 说明与限制
- 当前只实现了常用的 OpenAI 兼容子集，不是完整 OpenAI API。
- 每个请求目前只支持一张图片。
- 支持的消息角色为 `system`、`user`、`assistant`。
- `temperature` 参数会被接受，但当前适配器仍然使用贪心解码。
- 现有 CLI 推理与训练相关流程保持不变。
- 当前仓库中的 Qwen2-VL 路径存在既有运行时问题，本次改动不会修复这些问题。如果 Qwen2-VL 请求失败，应对现有适配器实现单独评估。

## 本功能未修改的仓库既有问题
- 文件：`SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`，第 22 行。
  问题：这里引用了 `configuration`，但当前方法里真正初始化的是 `self.configuration`，运行时可能触发 `NameError`。
  建议修改：将 `configuration._attn_implementation = "flash_attention_2"` 改为 `self.configuration._attn_implementation = "flash_attention_2"`。
- 文件：`SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`，第 33 行。
  问题：这里引用了 `lm_head`，但实例字段实际名为 `self.lm_head`，运行时可能触发 `NameError`。
  建议修改：将 `self.lm_head = lm_head.half().cuda(1)` 改为 `self.lm_head = self.lm_head.half().cuda(1)`。
- 文件：`SplitInfer/Models/Qwen2-VL-7B-Instruct/adapter.py`，第 59-60 行。
  问题：`hidden_states` 和 `position_ids` 保持在 `cuda:0`，而 `self.model_server` 与 `self.lm_head` 被放在 `cuda:1`，这可能导致跨设备运行时错误。
  建议修改：将这两个张量移动到与 `self.model_server` 相同的设备上，例如 `cuda(1)`。

## 常见问题
- `401 Unauthorized`：检查 `SPLITINFER_API_KEY` 和请求头中的 `Authorization`。
- `404 model_not_found`：`model` 必须与 `/v1/models` 中返回的模型名完全一致。
- `400 image_not_supported`：你把图片输入发给了纯文本模型。
- 如果报权重加载错误，通常表示 `SplitInfer/weights/` 下缺少对应模型文件。
- Qwen2-VL 调用失败也可能来自仓库现有问题，而不是 HTTP 服务层。
