"""Offline tests: use a tiny randomly initialised DINOv2 so nothing is downloaded."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import Dinov2Config, Dinov2Model  # noqa: E402

from models.dinov2_lora import build_model, count_params  # noqa: E402


def tiny_backbone():
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=64, image_size=28, patch_size=14)
    return Dinov2Model(cfg)


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_param_counts_ordered_linear_lt_lora_lt_full():
    counts = {m: trainable(build_model(tiny_backbone(), 5, mode=m)) for m in ("linear", "lora", "full")}
    assert counts["linear"] < counts["lora"] < counts["full"]


def test_linear_trains_only_head():
    model = build_model(tiny_backbone(), 5, mode="linear")
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_lora_params_are_adapters_and_head_only():
    model = build_model(tiny_backbone(), 5, mode="lora")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(("lora_" in n) or n.startswith("head") for n in names)


def test_forward_shape():
    model = build_model(tiny_backbone(), 5, mode="lora")
    out = model(torch.randn(2, 3, 28, 28))
    assert out.shape == (2, 5)


def test_lora_starts_at_pretrained_output():
    # B is initialised to zero, so a fresh LoRA model equals the frozen backbone.
    torch.manual_seed(0)
    base = tiny_backbone().eval()
    x = torch.randn(2, 3, 28, 28)
    ref = base(pixel_values=x).last_hidden_state
    model = build_model(base, 5, mode="lora").eval()
    out = model.backbone(pixel_values=x).last_hidden_state
    assert torch.allclose(ref, out, atol=1e-5)


def test_count_params_keys():
    stats = count_params(build_model(tiny_backbone(), 5, mode="full"))
    assert stats["trainable_pct"] == pytest.approx(100.0)
