import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    gpu: Optional[str] = None
    api_key: Optional[str] = None
    weights_root: Optional[str] = None
    model_paths: Dict[str, str] = field(default_factory=dict)


def parse_model_path_overrides(entries):
    overrides: Dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid --model-path value `{entry}`. Expected MODEL=PATH.")
        model_name, path = entry.split("=", 1)
        model_name = model_name.strip()
        path = path.strip()
        if not model_name or not path:
            raise ValueError(f"Invalid --model-path value `{entry}`. Expected MODEL=PATH.")
        overrides[model_name] = path
    return overrides


def load_server_settings(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ServerSettings:
    env_config_path = os.getenv("SPLITINFER_CONFIG")
    resolved_config_path = config_path or env_config_path

    config_data: Dict[str, Any] = {}
    if resolved_config_path:
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)

    model_paths = dict(config_data.get("model_paths") or {})
    env_model_paths = os.getenv("SPLITINFER_MODEL_PATHS_JSON")
    if env_model_paths:
        model_paths.update(json.loads(env_model_paths))

    settings = ServerSettings(
        host=str(config_data.get("host", "0.0.0.0")),
        port=int(config_data.get("port", 8000)),
        gpu=config_data.get("gpu") or os.getenv("SPLITINFER_GPU"),
        api_key=config_data.get("api_key") or os.getenv("SPLITINFER_API_KEY"),
        weights_root=config_data.get("weights_root") or os.getenv("SPLITINFER_WEIGHTS_ROOT"),
        model_paths=model_paths,
    )

    overrides = dict(cli_overrides or {})
    if overrides.get("host"):
        settings.host = overrides["host"]
    if overrides.get("port") is not None:
        settings.port = int(overrides["port"])
    if overrides.get("gpu"):
        settings.gpu = overrides["gpu"]
    if overrides.get("api_key"):
        settings.api_key = overrides["api_key"]
    if overrides.get("weights_root"):
        settings.weights_root = overrides["weights_root"]
    if overrides.get("model_paths"):
        settings.model_paths.update(overrides["model_paths"])

    return settings
