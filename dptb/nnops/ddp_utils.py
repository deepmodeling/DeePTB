import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from dptb.utils.argcheck import normalize
from dptb.utils.tools import j_loader


log = logging.getLogger(__name__)

DEPRECATED_TRAIN_OPTION_KEYS = (
    "shared_scheduler_metric",
    "independent_expert_scheduler",
    "distributed_global_reduce_every",
)

RESTART_LOCKED_TRAIN_OPTION_KEYS = (
    "optimizer",
    "lr_scheduler",
    "update_lr_per_iter",
    "distance_ranges",
    "expert_lrs",
    "expert_optimizer_overrides",
    "expert_lr_scheduler_overrides",
)


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def derive_rank_log_path(log_path: Optional[str], rank: int) -> Optional[str]:
    if log_path is None:
        return None
    p = Path(log_path)
    if rank == 0:
        return str(p)
    suffix = p.suffix if p.suffix else ".txt"
    return str(p.with_name(f"{p.stem}.rank{rank}{suffix}"))


def destroy_process_group_safely() -> None:
    if is_dist_ready():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def get_current_cuda_device_index() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


def init_process_group_with_device(
    *,
    backend: str,
    rank: int,
    world_size: int,
    timeout,
) -> None:
    kwargs = dict(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    current_device = get_current_cuda_device_index()
    if current_device is not None:
        try:
            dist.init_process_group(
                device_id=torch.device(f"cuda:{current_device}"),
                **kwargs,
            )
            return
        except TypeError:
            pass
    dist.init_process_group(**kwargs)


def dist_barrier_on_current_device() -> None:
    if not is_dist_ready():
        return
    current_device = get_current_cuda_device_index()
    if current_device is not None:
        try:
            dist.barrier(device_ids=[current_device])
            return
        except TypeError:
            pass
    dist.barrier()


def strip_deprecated_train_options(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    train_opt = raw_config.get("train_options", None)
    if not isinstance(train_opt, dict):
        return raw_config

    removed = {k: train_opt.pop(k) for k in DEPRECATED_TRAIN_OPTION_KEYS if k in train_opt}
    if removed:
        removed_keys = ", ".join(sorted(removed.keys()))
        log.warning(
            "Ignoring deprecated train_options keys: %s. "
            "Supported per-expert LR interfaces are `expert_lrs`, "
            "`expert_optimizer_overrides`, and `expert_lr_scheduler_overrides`.",
            removed_keys,
        )
    return raw_config


def load_multi_train_config(input_path: str) -> Dict[str, Any]:
    raw = j_loader(input_path)
    raw = strip_deprecated_train_options(raw)
    return normalize(raw)


def configure_debug_env(train_opt: Dict[str, Any]) -> None:
    if bool(train_opt.get("ddp_debug_detail", False)):
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

    if bool(train_opt.get("nccl_debug", False)):
        os.environ.setdefault("NCCL_DEBUG", str(train_opt.get("nccl_debug_level", "INFO")))

    if bool(train_opt.get("cuda_launch_blocking", False)):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if bool(train_opt.get("nccl_async_error_handling", True)):
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def configure_runtime_perf(train_opt: Dict[str, Any]) -> None:
    cudnn_benchmark = bool(train_opt.get("cudnn_benchmark", False))
    allow_tf32 = bool(train_opt.get("allow_tf32", True))
    matmul_precision = str(train_opt.get("float32_matmul_precision", "")).strip()

    try:
        torch.backends.cudnn.benchmark = cudnn_benchmark
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except Exception:
            pass

    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass

    log.info(
        "[runtime] cudnn_benchmark=%s, allow_tf32=%s, float32_matmul_precision=%s",
        cudnn_benchmark,
        allow_tf32,
        matmul_precision or "default",
    )


def merge_restart_train_options(
    requested_train_options: Optional[Dict[str, Any]],
    ckpt_train_options: Optional[Dict[str, Any]],
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or log
    ckpt_train_options = copy.deepcopy(ckpt_train_options or {})
    merged_train_options = copy.deepcopy(ckpt_train_options)
    merged_train_options.update(copy.deepcopy(requested_train_options or {}))

    for key in RESTART_LOCKED_TRAIN_OPTION_KEYS:
        if key not in ckpt_train_options:
            continue
        if merged_train_options.get(key) != ckpt_train_options.get(key):
            logger.warning(
                "%s in config file is not consistent with the checkpoint, using the one in checkpoint",
                key,
            )
        merged_train_options[key] = copy.deepcopy(ckpt_train_options[key])

    return merged_train_options
