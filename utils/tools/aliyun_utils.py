"""
Aliyun LLM client: DashScope API with multi-turn chat and OpenAI-compatible interface.
Config is loaded from YAML (path resolved relative to project root).
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from openai import OpenAI


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "aliyun_config.yaml"


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file; resolve relative paths against project root."""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def set_aliyun_config(
    api_key: str,
    region: str,
    base_url: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Programmatically set Aliyun LLM config in a YAML file.

    This updates (or creates) a config file with:

        aliyun:
          api_key: ...
          region: ...
          base_url: ...

    Other top-level keys and sections are preserved.
    Returns the resolved config path.
    """
    if not api_key:
        raise ValueError("api_key must be non-empty")
    if not region:
        raise ValueError("region must be non-empty")
    if not base_url:
        raise ValueError("base_url must be non-empty")

    cfg_path = _resolve_path(config_path or _DEFAULT_CONFIG_PATH)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    aliyun_cfg = data.get("aliyun") or {}
    aliyun_cfg.update(
        {
            "api_key": api_key,
            "region": region,
            "base_url": base_url,
        }
    )
    data["aliyun"] = aliyun_cfg

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

    return cfg_path


class AliyunLLMClient:
    """Aliyun LLM client for multi-turn conversations."""

    @classmethod
    def _load_config(cls, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load and validate aliyun section from YAML. Requires aliyun.api_key, region, base_url."""
        data = _load_yaml(config_path or _DEFAULT_CONFIG_PATH)
        cfg = data.get("aliyun")
        if not cfg:
            raise ValueError("Config must contain 'aliyun' section.")
        for key in ("api_key", "region", "base_url"):
            if key not in cfg:
                raise ValueError(f"Config missing 'aliyun.{key}'.")
        return cfg

    @classmethod
    def get_prompt_template(cls, prompt_name: str, config_path: Optional[Union[str, Path]] = None) -> str:
        """Return prompt template string for prompt_name from config aliyun.prompts."""
        cfg = cls._load_config(config_path)
        prompts = cfg.get("prompts") or {}
        if prompt_name not in prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found. Available: {list(prompts.keys())}")
        return prompts[prompt_name]

    def __init__(self, model: str = "qwen3-max", config_path: Optional[Union[str, Path]] = None):
        """Initialize client; load api_key, region, base_url from YAML. model default: qwen3-max."""
        cfg = self._load_config(config_path)
        self.api_key = cfg["api_key"]
        self.region = cfg["region"]
        self.base_url = cfg["base_url"]
        if not self.api_key:
            raise ValueError(
                "Set 'aliyun.api_key' in config. See: https://help.aliyun.com/zh/model-studio/get-api-key"
            )
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Return assistant text for messages [{'role': '...', 'content': '...'}, ...]."""
        try:
            completion = self.client.chat.completions.create(model=self.model, messages=messages)
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Model {self.model} call failed: {e}") from e
