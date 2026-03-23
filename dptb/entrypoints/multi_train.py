import os
import json
import time
import heapq
import logging
import sys
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from dptb.plugins.monitor import (
    TrainLossMonitor, LearningRateMonitor, Validationer, TensorBoardMonitor,
    DeepDoctorMonitor, SO2ModuleMonitor, PreTPBlockMonitor,
    TrainOnsiteLossMonitor, TrainHoppingLossMonitor, TrainZLossMonitor, ExpertLoadCVMonitor,
    ScalarFieldMonitor
)
from dptb.plugins.train_logger import Logger
from dptb.plugins.saver import Saver
from dptb.utils.argcheck import normalize, collect_cutoffs, chk_avg_per_iter
from dptb.utils.tools import j_loader, setup_seed, j_must_have
from dptb.utils.loggers import set_log_handles

from dptb.entrypoints.train import deep_dict_difference, print_model_params_detailed
from dptb.nnops.multi_trainer import MultiTrainer

__all__ = ["multi_train"]

log = logging.getLogger(__name__)


# --------------------------- TAGGER ---------------------------
class _EntryTagger:
    def __init__(self, enabled: bool, cuda_mem: bool, cuda_sync: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)
        self.device = device

    def _cuda_mem(self):
        if not (torch.cuda.is_available() and self.device.type == "cuda"):
            return None
        alloc = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None: return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return (f" | cuda_alloc={alloc / mb:.1f}MB cuda_reserved={reserved / mb:.1f}MB "
                f"cuda_peak={peak / mb:.1f}MB free={free / mb:.1f}MB total={total / mb:.1f}MB")

    @contextlib.contextmanager
    def tag(self, name: str, extra: str = ""):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        if self.cuda_mem and self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except:
                pass

        try:
            yield
        finally:
            if self.cuda_sync and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem() if (self.cuda_mem and self.device.type == "cuda") else None
            log.info(f"[TAG][ENTRY][{name}] dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")


# --------------------------- 参数统计 ---------------------------
def _format_params_lazy(num: int) -> str:
    if num >= 1_000_000_000: return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000: return f"{num / 1_000_000:.2f}M"
    if num >= 1_000: return f"{num / 1_000:.2f}K"
    return str(num)


def _count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def print_multi_model_params_detailed(model: nn.Module, logger=None, max_depth: int = 5):
    log_func = logger.info if logger else print
    actual_model = model.module if isinstance(model, DDP) else model

    if not hasattr(actual_model, "experts") or not isinstance(actual_model.experts, nn.ModuleList) or len(
            actual_model.experts) == 0:
        print_model_params_detailed(actual_model, logger=logger, max_depth=max_depth)
        return

    expert_stats = [_count_params(expert) for expert in actual_model.experts]
    single_stat = expert_stats[0]
    experts_sum = {"total": sum(x["total"] for x in expert_stats),
                   "trainable": sum(x["trainable"] for x in expert_stats)}
    wrapper_stat = _count_params(actual_model)

    log_func("=" * 80)
    log_func(f"MULTI-EXPERT PARAMETER SUMMARY (Experts: {len(expert_stats)})")
    log_func("-" * 80)
    log_func(
        f"Single Expert (expert_0): total={_format_params_lazy(single_stat['total'])}, trainable={_format_params_lazy(single_stat['trainable'])}")
    log_func(
        f"All Experts Sum:          total={_format_params_lazy(experts_sum['total'])}, trainable={_format_params_lazy(experts_sum['trainable'])}")
    log_func(
        f"Wrapper Total:            total={_format_params_lazy(wrapper_stat['total'])}, trainable={_format_params_lazy(wrapper_stat['trainable'])}")
    log_func("=" * 80)
    print_model_params_detailed(actual_model.experts[0], logger=logger, max_depth=max_depth)


# --------------------------- 核心训练入口 ---------------------------
def multi_train(
        INPUT: str,
        init_model: Optional[str],
        restart: Optional[str],
        output: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    # ==================== DDP 强力嗅探器 ====================
    # 无论用户从哪里启动代码，只要环境里有 DDP 变量，立刻识别身份并分配显卡
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        _DDP_RANK = int(os.environ["RANK"])
        _DDP_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        _DDP_AVAILABLE = True
        _DDP_LOCAL_RANK = int(os.environ["LOCAL_RANK"])

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=_DDP_RANK, world_size=_DDP_WORLD_SIZE)
            log.info(
                f"DDP Init: Process group initialized. Rank {_DDP_RANK}/{_DDP_WORLD_SIZE}, Local GPU: {_DDP_LOCAL_RANK}")
    else:
        _DDP_RANK, _DDP_WORLD_SIZE, _DDP_AVAILABLE, _DDP_LOCAL_RANK = 0, 1, False, 0

    run_opt: Dict[str, Any] = {"init_model": init_model, "restart": restart, "log_path": log_path,
                               "log_level": log_level}

    # ==================== 目录与日志控制 ====================
    if output:
        if _DDP_RANK == 0:  # 只有主进程建文件夹
            Path(output).mkdir(exist_ok=True, parents=True)
            checkpoint_path = os.path.join(str(output), "checkpoint")
            Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
            if not log_path: log_path = os.path.join(str(output), "log/log.txt")
            Path(log_path).parent.mkdir(exist_ok=True, parents=True)
            run_opt.update(
                {"output": str(Path(output).absolute()), "checkpoint_path": str(Path(checkpoint_path).absolute()),
                 "log_path": str(Path(log_path).absolute())})

        if _DDP_AVAILABLE:
            dist.barrier()  # 等待主进程建好目录
            if _DDP_RANK != 0:
                run_opt.update({"output": None, "checkpoint_path": None, "log_path": None})

    set_log_handles(log_level, Path(run_opt["log_path"]) if run_opt["log_path"] else None)

    # 从进程静音，保持控制台整洁
    if _DDP_AVAILABLE and _DDP_RANK != 0:
        log.setLevel(logging.WARNING)

    # 读取 JSON 配置
    jdata = normalize(j_loader(INPUT))

    # ==================== 设备分配 ====================
    if _DDP_AVAILABLE:
        jdata["common_options"]["device"] = f"cuda:{_DDP_LOCAL_RANK}"
        torch.cuda.set_device(jdata["common_options"]["device"])

    entry_device = torch.device(jdata["common_options"]["device"])
    dbg = jdata.get("train_options", {})
    entry_tagger = _EntryTagger(
        enabled=bool(dbg.get("debug_tags", False)),
        cuda_mem=bool(dbg.get("debug_tag_cuda_mem", True)),
        cuda_sync=bool(dbg.get("debug_tag_cuda_sync", False)),
        device=entry_device
    )

    with entry_tagger.tag("set_default_dtype"):
        torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    with entry_tagger.tag("merge_config"):
        if restart or init_model:
            f = restart if restart else init_model
            f_data = torch.load(f, map_location=entry_device, weights_only=False)
            if jdata.get("model_options", None) is None: jdata["model_options"] = f_data["config"]["model_options"]
            if jdata.get("train_options", None) is None: jdata["train_options"] = f_data["config"]["train_options"]
            del f_data

    with entry_tagger.tag("setup_seed"):
        # DDP 训练时，必须保证每个进程的 Seed 不同，否则 DataLoader 抽出的 Batch 是一模一样的
        setup_seed(seed=jdata["common_options"]["seed"] + _DDP_RANK)

    cutoff_options = collect_cutoffs(jdata)

    with entry_tagger.tag("build_dataset"):
        train_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["train"], **jdata["common_options"])
        validation_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["validation"],
                                            **jdata["common_options"]) if jdata["data_options"].get(
            "validation") else None
        reference_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["reference"],
                                           **jdata["common_options"]) if jdata["data_options"].get(
            "reference") else None

    distance_ranges = jdata["train_options"].get("distance_ranges", [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]])

    # ==================== 构建模型与 DDP 包裹 ====================
    if restart:
        with entry_tagger.tag("trainer/restart"):
            trainer = MultiTrainer.restart(
                checkpoint=restart, train_datasets=train_datasets, train_options=jdata["train_options"],
                common_options=jdata["common_options"], reference_datasets=reference_datasets,
                validation_datasets=validation_datasets,
                ddp_rank=_DDP_RANK, ddp_world_size=_DDP_WORLD_SIZE, ddp_available=_DDP_AVAILABLE,
            )
        model = trainer.model.to(entry_device)
    else:
        with entry_tagger.tag("build_model"):
            model = build_model(checkpoint=init_model, model_options=jdata["model_options"],
                                common_options=jdata["common_options"], train_options=jdata["train_options"])
        model.to(entry_device)

    # DDP 包裹
    if _DDP_AVAILABLE:
        log.info(f"DDP: Wrapping model for rank {_DDP_RANK} on {entry_device}")
        model = DDP(model, device_ids=[_DDP_LOCAL_RANK])

    if restart and _DDP_AVAILABLE:
        trainer.model = model

    scale_type = jdata["model_options"]["prediction"].get('scale_type', "scale_w_back_grad")
    if scale_type != 'no_scale':
        with entry_tagger.tag("dataset/E3statistics"):
            # 只有 rank 0 算归一化，DDP 模型需要解包 module
            if _DDP_RANK == 0: train_datasets.E3statistics(model=model.module if _DDP_AVAILABLE else model)
            if _DDP_AVAILABLE: dist.barrier()

    if not restart:
        with entry_tagger.tag("trainer/init"):
            trainer = MultiTrainer(
                distance_ranges=distance_ranges, train_options=jdata["train_options"],
                common_options=jdata["common_options"],
                model=model, train_datasets=train_datasets, validation_datasets=validation_datasets,
                reference_datasets=reference_datasets,
                ddp_rank=_DDP_RANK, ddp_world_size=_DDP_WORLD_SIZE, ddp_available=_DDP_AVAILABLE,
            )

    # ==================== 注册插件 (严格防踩踏) ====================
    with entry_tagger.tag("trainer/register_plugins"):
        log_field = ["train_loss", "lr"]
        if validation_datasets:
            trainer.register_plugin(
                Validationer(interval=[(jdata["train_options"]["validation_freq"], 'iteration'), (1, 'epoch')],
                             fast_mode=jdata["train_options"]["valid_fast"]))
            log_field.append("validation_loss")

        trainer.register_plugin(TrainLossMonitor(sliding_win_size=jdata["train_options"]["sliding_win_size"],
                                                 avg_per_iter=chk_avg_per_iter(jdata)))
        trainer.register_plugin(LearningRateMonitor())
        trainer.register_plugin(TrainOnsiteLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(TrainHoppingLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(TrainZLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(ExpertLoadCVMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(
            ScalarFieldMonitor(stat_name="train_loss_opt", interval=[(1, 'iteration'), (1, 'epoch')]))

        log_field.extend(["mean_max_prob", "expert_load_cv", "train_onsite_loss", "train_hopping_loss"])

        # ======= 只有主进程 (Rank 0) 允许注册写入文件的插件 =======
        if _DDP_RANK == 0:
            if jdata["train_options"].get("monitor_flag"):
                trainer.register_plugin(DeepDoctorMonitor(output, verbose_freq=1))
                trainer.register_plugin(SO2ModuleMonitor(output))
                trainer.register_plugin(PreTPBlockMonitor(output))
            if jdata["train_options"].get("use_tensorboard"):
                trainer.register_plugin(
                    TensorBoardMonitor(interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')],
                                       log_dir=os.path.join(output, "tensorboard_logs") if output else "./tb_logs"))
            if run_opt["output"]:
                with open(os.path.join(run_opt["output"], "train_config.json"), "w") as fp: json.dump(jdata, fp,
                                                                                                      indent=4)
            # 极其重要：只有主进程添加 Saver！这样从进程在 epoch 结束时绝对不会执行 Saver，解决文件竞争
            if run_opt["checkpoint_path"] and jdata["train_options"].get("save_freq"):
                # 注意：使用你项目原版的 Saver 参数签名
                trainer.register_plugin(
                    Saver(interval=[(jdata["train_options"].get("save_freq"), 'iteration'), (1, 'epoch')]),
                    checkpoint_path=run_opt["checkpoint_path"])

        trainer.register_plugin(
            Logger(log_field, interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))
        for q in trainer.plugin_queues.values(): heapq.heapify(q)

    print_multi_model_params_detailed(trainer.model, logger=log, max_depth=5)

    with entry_tagger.tag("trainer/run"):
        start_time = time.time()
        trainer.run(trainer.train_options["num_epoch"])
        end_time = time.time()

    if _DDP_AVAILABLE: dist.destroy_process_group()
    if _DDP_RANK == 0:
        log.info("finished training")
        log.info(f"wall time: {(end_time - start_time):.3f} s")