"""Fine-tune DINOv2 on dataset of choice with MLflow tracking.

Example:
    docker compose run --rm <name> python -m train.train --mode lora --epochs 3
"""
from __future__ import annotations

import argparse
import os
import time

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import build_datasets
from train.transforms import (
    TransformedDataset,
    build_eval_transform,
    build_train_transform,
)
from models.dinov2_lora import MODES, build_model, count_params, load_backbone
from train.hyperparameters import DATA_PARAMS, DEFAULTS, DEFAULT_LR
from utils import git_commit, resolve_device, seed_everything, worker_init_fn


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line options for training.

    :param args, list[str] | None; Arguments to parse, or None to use sys.argv.
    :return: Parsed training options.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default=DEFAULTS["mode"])
    p.add_argument("--backbone", default=DEFAULTS["backbone"])
    p.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    p.add_argument("--val-frac", type=float, default=DEFAULTS["val_frac"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=None, help="defaults depend on --mode")
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    p.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    p.add_argument("--lora-dropout", type=float, default=DEFAULTS["lora_dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="debug: use only the first N training samples")
    p.add_argument("--experiment", default="dinov2-pets")
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////workspace/mlflow.db"),
    )
    return p.parse_args(args)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a model and return its average loss and accuracy.

    :param model, nn.Module; Model to evaluate.
    :param loader, DataLoader; Batches of input tensors and class labels.
    :param device, torch.device; Device on which to run evaluation.
    :return: Mapping containing average loss and accuracy.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return {"loss": total_loss / n, "acc": correct / n}


def train_one_epoch(model, loader, optimizer, device) -> dict[str, float]:
    """Train the model for one epoch and report average loss and accuracy.

    :param model, nn.Module; Model whose trainable parameters will be updated.
    :param loader, DataLoader; Batches of input tensors and class labels.
    :param optimizer, torch.optim.Optimizer; Optimizer used to update parameters.
    :param device, torch.device; Device on which to run training.
    :return: Mapping containing average loss and accuracy.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return {"loss": total_loss / n, "acc": correct / n}


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the state of parameters that were trained.

    For LoRA, this contains the adapters and classifier head rather than the full
    backbone.

    :param model, nn.Module; Model whose trainable parameter state to extract.
    :return: CPU copies of trainable parameters, keyed by state-dict name.
    """
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def main(argv=None) -> dict[str, float]:
    """Train, select the best validation checkpoint, and optionally test it.

    :param argv, list[str] | None; Command-line arguments, or None to use sys.argv.
    :return: Best validation accuracy, and test accuracy if test evaluation is enabled.
    """
    args = parse_args(argv)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]
    args.lr = lr
    seed_everything(args.seed, deterministic=not args.no_deterministic)
    device = resolve_device(args.device)

    mlflow.set_tracking_uri(args.tracking_uri)
    train_ds, val_ds, test_ds, info = build_datasets(
        args.dataset_name, args.val_frac, args.seed
    )
    train_ds = TransformedDataset(train_ds, build_train_transform())
    val_ds = TransformedDataset(val_ds, build_eval_transform())
    test_ds = TransformedDataset(test_ds, build_eval_transform())
    if args.max_train_samples:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))

    # A seeded generator makes the shuffling order reproducible.
    gen = torch.Generator().manual_seed(args.seed)
    common = dict(num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=gen, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, **common)

    model = build_model(
        load_backbone(args.backbone), info["num_classes"], mode=args.mode,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)
    params = count_params(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay,
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.mode}-r{args.lora_r}" if args.mode == "lora" else args.mode):
        mlflow.log_params({
            **{k: v for k, v in vars(args).items() if k != "tracking_uri"},
            "git_commit": git_commit(),
            "dataset": args.dataset_name,
            "split_seed": args.seed,
            "split_val_frac": args.val_frac,
            "image_size": DATA_PARAMS["image_size"],
            "eval_resize_scale": DATA_PARAMS["eval_resize_scale"],
            "image_mean": str(DATA_PARAMS["imagenet_mean"]),
            "image_std": str(DATA_PARAMS["imagenet_std"]),
            "train_crop_scale": str(DATA_PARAMS["train_crop_scale"]),
            "train_horizontal_flip_prob": DATA_PARAMS["train_horizontal_flip_prob"],
            "n_train": info["n_train"], "n_val": info["n_val"], "n_test": info["n_test"],
            "device": str(device),
            "torch_version": torch.__version__,
        })
        mlflow.log_metrics(params)

        best_val, best_state, start = -1.0, None, time.time()
        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device)
            va = evaluate(model, val_loader, device)
            mlflow.log_metrics({"train_loss": tr["loss"], "train_acc": tr["acc"],
                                "val_loss": va["loss"], "val_acc": va["acc"]}, step=epoch)
            print(f"epoch {epoch}/{args.epochs}  train_acc={tr['acc']:.3f}  val_acc={va['acc']:.3f}")
            if va["acc"] > best_val:
                best_val, best_state = va["acc"], trainable_state_dict(model)
        train_seconds = time.time() - start

        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if set(best_state) != trainable_names:
            raise RuntimeError("Best checkpoint does not contain every trainable parameter")
        load_result = model.load_state_dict(best_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Best checkpoint contains unexpected keys: {load_result.unexpected_keys}"
            )
        missing_trainable = trainable_names.intersection(load_result.missing_keys)
        if missing_trainable:
            raise RuntimeError(
                f"Best checkpoint failed to restore trainable keys: "
                f"{sorted(missing_trainable)}"
            )

        mlflow.log_metrics({"best_val_acc": best_val, "train_seconds": train_seconds})
        test = evaluate(model, test_loader, device)
        mlflow.log_metrics({"test_acc": test["acc"], "test_loss": test["loss"]})

        os.makedirs("outputs", exist_ok=True)
        ckpt = f"outputs/{args.mode}_trainable.pt"
        torch.save(best_state, ckpt)
        mlflow.log_artifact(ckpt, artifact_path="checkpoint")
        mlflow.log_metric("checkpoint_mb", os.path.getsize(ckpt) / 1e6)
        print(f"test_acc={test['acc']:.4f}  trainable={params['trainable_params']:,} "
              f"({params['trainable_pct']:.2f}%)  time={train_seconds:.0f}s")
    return {"test_acc": test["acc"], "best_val_acc": best_val}


if __name__ == "__main__":
    main()
