"""Default hyperparameters for DINOv2 fine-tuning."""

DEFAULTS = {
    "mode": "lora",
    "backbone": "facebook/dinov2-small",
    "dataset_name": "timm/oxford-iiit-pet",
    "epochs": 3,
    "batch_size": 32,
    "weight_decay": 0.01,
    "val_frac": 0.15,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "seed": 42,
    "num_workers": 2,
}

DEFAULT_LR = {
    "linear": 1e-3,
    "lora": 5e-4,
    "full": 2e-5,
}

DATA_PARAMS = {
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
    "image_size": 224,
    "eval_resize_scale": 256 / 224,
    "train_crop_scale": (0.7, 1.0),
    "train_horizontal_flip_prob": 0.5,
}
