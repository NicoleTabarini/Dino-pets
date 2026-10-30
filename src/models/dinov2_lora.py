"""DINOv2 classifier with three fine-tuning modes: linear, LoRA, full."""
from __future__ import annotations

import torch
from torch import nn

MODES = ("linear", "lora", "full")


class DinoClassifier(nn.Module):
    """Backbone -> CLS token -> linear head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int):
        """Initialize a DINOv2 backbone and classification head.

        :param backbone, nn.Module; DINOv2 image encoder.
        :param hidden_size, int; Feature size emitted by the backbone.
        :param num_classes, int; Number of output classes.
        :return: None.
        """
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Classify a batch of normalized image tensors.

        :param pixel_values, torch.Tensor; Batch of images shaped (N, C, H, W).
        :return: Class logits shaped (N, num_classes).
        """
        # last_hidden_state[:, 0] is the CLS token: a global summary of the image.
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(cls)


def load_backbone(name: str = "facebook/dinov2-small") -> nn.Module:
    """Load a pretrained DINOv2 backbone.

    :param name, str; Hugging Face model identifier or local model path.
    :return: Loaded DINOv2 model.
    """
    from transformers import Dinov2Model

    return Dinov2Model.from_pretrained(name)


def build_model(
    backbone: nn.Module,
    num_classes: int,
    mode: str = "lora",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> DinoClassifier:
    """Build a classifier configured for the requested fine-tuning mode.

    In linear mode, the backbone is frozen and only the head is trained. LoRA mode
    injects low-rank adapters into attention query and value projections, while
    full mode trains the complete model.

    :param backbone, nn.Module; Pretrained image encoder to wrap.
    :param num_classes, int; Number of output classes.
    :param mode, str; Fine-tuning mode: "linear", "lora", or "full".
    :param lora_r, int; LoRA rank used when mode is "lora".
    :param lora_alpha, int; LoRA scaling factor used when mode is "lora".
    :param lora_dropout, float; Dropout probability for LoRA layers.
    :return: Configured DINOv2 classifier.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    hidden = backbone.config.hidden_size

    if mode == "linear":
        for p in backbone.parameters():
            p.requires_grad = False
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
        )
        backbone = get_peft_model(backbone, cfg)  # freezes the base weights

    return DinoClassifier(backbone, hidden, num_classes)


def count_params(model: nn.Module) -> dict[str, float]:
    """Count total and trainable model parameters.

    :param model, nn.Module; Model whose parameters will be counted.
    :return: Mapping containing total count, trainable count, and trainable percent.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total,
    }
