"""Save/load model state; load uses CellSTIC and config from model.train."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import CellSTICConfig


class ModelUtils:
    """Save / load ``cellstic_model.pth`` only (no auxiliary graph files)."""

    @staticmethod
    def save_model(model: nn.Module, model_path: str) -> None:
        """Save model.state_dict() to model_path (dir) / cellstic_model.pth."""
        save_dir = Path(model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "cellstic_model.pth")

    @staticmethod
    def load_model(
        model_path: str,
        config: CellSTICConfig,
        device: torch.device,
        hierarchy_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Load state_dict into CellSTIC; optionally init head layers from ``hierarchy_dict``."""
        from model import CellSTIC

        model_dir = Path(model_path)
        model_file = model_dir / "cellstic_model.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                f"No model file found in {model_dir} (expected cellstic_model.pth or signal_logic_model.pth)"
            )
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        model = CellSTIC(config.model, device)
        if hierarchy_dict is not None:
            try:
                model.ccc_predictor.init_head_layers(hierarchy_dict)
            except Exception:
                pass
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        return model
