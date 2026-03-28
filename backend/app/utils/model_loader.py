from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from app.models.survival_net import SurvivalNet
from app.models.unet_light import UNetLight

logger = logging.getLogger("model_loader")


@dataclass(frozen=True)
class Models:
    segmenter: UNetLight
    survival: SurvivalNet


_MODELS: Models | None = None


def _try_load_weights(model: torch.nn.Module, path: str) -> None:
    if not path:
        return
    if not os.path.exists(path):
        logger.info("No weights found at %s (using random init).", path)
        return
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", path)
    except Exception:
        logger.exception("Failed loading weights from %s (using random init).", path)


def get_models() -> Models:
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    seg = UNetLight(in_channels=4, out_channels=3, base=16)
    surv = SurvivalNet()

    seg_path = os.getenv("SEGMENTATION_WEIGHTS", "unet_trained.pth")
    surv_path = os.getenv("SURVIVAL_WEIGHTS", os.path.join("models", "survival_net.pt"))
    _try_load_weights(seg, seg_path)
    _try_load_weights(surv, surv_path)
    seg.eval()
    surv.eval()

    for m in (seg, surv):
        for p in m.parameters():
            p.requires_grad_(False)

    _MODELS = Models(segmenter=seg, survival=surv)
    return _MODELS

