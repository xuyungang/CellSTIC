"""Save/load model state; load uses CellSTIC and config from utils.train."""

from pathlib import Path

import pickle
import torch
import torch.nn as nn

from utils.train.config import ModelConfig


class ModelUtils:
    """Save model state_dict; load into CellSTIC, init head_layers from model_dir / hierarchy_dict.pkl if present."""

    @staticmethod
    def save_model(model: nn.Module, model_path: str) -> None:
        """Save model.state_dict() to model_path (dir) / cellstic_model.pth."""
        save_dir = Path(model_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "cellstic_model.pth")

    @staticmethod
    def load_model(
        model_path: str,
        config: ModelConfig,
        device: torch.device,
    ) -> nn.Module:
        """Load state_dict into CellSTIC; init head_layers from model_dir / hierarchy_dict.pkl if present."""
        from model import CellSTIC

        model_dir = Path(model_path)
        model_file = model_dir / "cellstic_model.pth"
        if not model_file.exists():
            legacy = model_dir / "signal_logic_model.pth"
            if legacy.exists():
                model_file = legacy
        if not model_file.exists():
            raise FileNotFoundError(
                f"No model file found in {model_dir} (expected cellstic_model.pth, signal_logic_model.pth, lrst_ccc_model.pth, or spagem_model.pth)"
            )
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        model_config = getattr(config, "model", config)
        model = CellSTIC(model_config, device)
        if (model_dir / "hierarchy_dict.pkl").exists():
            try:
                with open(model_dir / "hierarchy_dict.pkl", "rb") as f:
                    hierarchy_dict = pickle.load(f)
            except Exception:
                hierarchy_dict = None
        else:
            hierarchy_dict = None
        if hierarchy_dict is not None:
            try:
                model.ccc_predictor.init_head_layers(hierarchy_dict)
            except Exception:
                pass
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        return model
