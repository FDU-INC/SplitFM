import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional


MODEL_REGISTRY = {
    "Qwen2-VL-7B-Instruct": {
        "folder": "Qwen2-VL-7B-Instruct",
        "adapter_class": "QwenAdapter",
        "supports_images": True,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "folder": "DeepSeek-R1-Distill-Llama-8B",
        "adapter_class": "DeepSeekAdapter",
        "supports_images": False,
    },
    "Llama-3-8B-Instruct": {
        "folder": "Llama-3-8B-Instruct",
        "adapter_class": "LlamaAdapter",
        "supports_images": False,
    },
}


def get_base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_default_weights_root() -> str:
    return os.path.join(get_base_dir(), "weights")


def get_weights_path(
    model_name: str,
    weights_root: Optional[str] = None,
    model_paths: Optional[Dict[str, str]] = None,
) -> str:
    if model_paths and model_name in model_paths:
        return model_paths[model_name]

    resolved_root = weights_root or get_default_weights_root()
    return os.path.join(resolved_root, model_name)


def get_model_config(model_name: str) -> Dict[str, object]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported models: {model_name}.")
    return MODEL_REGISTRY[model_name]


def load_adapter(
    model_name: str,
    weights_root: Optional[str] = None,
    model_paths: Optional[Dict[str, str]] = None,
):
    config = get_model_config(model_name)

    base_dir = get_base_dir()
    code_dir = os.path.join(base_dir, "Models", config["folder"])
    adapter_path = os.path.join(code_dir, "adapter.py")
    weights_path = get_weights_path(model_name, weights_root=weights_root, model_paths=model_paths)

    module_name = f"splitinfer_{config['folder'].replace('-', '_')}_adapter"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load adapter module from {adapter_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    adapter_class = getattr(module, config["adapter_class"])
    adapter = adapter_class(model_name)
    return adapter, weights_path


@dataclass
class LoadedAdapter:
    model_name: str
    adapter: object
    weights_path: str

    @property
    def supports_images(self) -> bool:
        return bool(MODEL_REGISTRY[self.model_name]["supports_images"])

    @property
    def tokenizer(self):
        return getattr(self.adapter, "tokenizer", None)


class ModelManager:
    def __init__(
        self,
        weights_root: Optional[str] = None,
        model_paths: Optional[Dict[str, str]] = None,
    ):
        self._cache: Dict[str, LoadedAdapter] = {}
        self._lock = threading.Lock()
        self.weights_root = weights_root
        self.model_paths = dict(model_paths or {})

    def get(self, model_name: str) -> LoadedAdapter:
        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]

            adapter, weights_path = load_adapter(
                model_name,
                weights_root=self.weights_root,
                model_paths=self.model_paths,
            )
            adapter.load(weights_path)
            loaded = LoadedAdapter(
                model_name=model_name,
                adapter=adapter,
                weights_path=weights_path,
            )
            self._cache[model_name] = loaded
            return loaded

    def list_models(self):
        return list(MODEL_REGISTRY.keys())
