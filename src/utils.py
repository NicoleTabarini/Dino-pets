"""Reproducibility helpers: seeding, determinism flags, git info, device choice."""
from __future__ import annotations

import os
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed random number generators and optionally request deterministic kernels.

    Deterministic mode costs some speed and gives repeatable results on the SAME
    hardware and library versions. It does not promise identical numbers across
    different GPUs or versions; pinning them in the container is what covers that.

    :param seed, int; Seed applied to Python, NumPy, and PyTorch random generators.
    :param deterministic, bool; Whether to enable deterministic PyTorch algorithms.
    :return: None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker.

    :param worker_id, int; Worker identifier supplied by the DataLoader.
    :return: None.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def git_commit() -> str:
    """Get the current Git commit and mark it if the worktree is dirty.

    :return: Commit SHA, optionally suffixed with "-dirty", or "unknown" if Git
        metadata cannot be read.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device name or select the best available accelerator.

    :param name, str; Explicit PyTorch device string, or "auto" for automatic
        selection.
    :return: Selected PyTorch device.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
