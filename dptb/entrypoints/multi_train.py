import os
import json
import time
import heapq
import logging
import sys
import contextlib
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

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


class _EntryTagger:
    def __init__(self, enabled: bool, cuda_mem: bool, cuda_sync: bool):
        self.enabled = bool(enabled)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)

    def _cuda_mem(self, device: torch.device):
        if not (torch.cuda.is_available() and device.type == "cuda"):
            return None
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        free, total = torch.cuda.mem_get_info(device)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None:
            return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return (f" | cuda_alloc={alloc/mb:.1f}MB cuda_reserved={reserved/mb:.1f}MB "
                f"cuda_peak={peak/mb:.1f}MB free={free/mb:.1f}MB total={total/mb:.1f}MB")

    @contextlib.contextmanager
    def tag(self, name: str, device: Optional[torch.device] = None, extra: str = ""):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()

        dev = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.cuda_mem and dev.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        try:
            yield
        finally:
            if self.cuda_sync and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem(dev) if (self.cuda_mem and dev.type == "cuda") else None
            log.info(f"[TAG][ENTRY][{name}] dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")


def _format_params_lazy(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def _count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {"total": total, "trainable": trainable, "non_trainable": non_trainable}


def print_multi_model_params_detailed(model: nn.Module, logger=None, max_depth: int = 5):
    log_func = logger.info if logger else print

    if not hasattr(model, "experts") or not isinstance(model.experts, nn.ModuleList) or len(model.experts) == 0:
        print_model_params_detailed(model, logger=logger, max_depth=max_depth)
        return

    expert_stats = [_count_params(expert) for expert in model.experts]
    num_experts = len(expert_stats)

    single_stat = expert_stats[0]
    experts_sum = {
        "total": sum(x["total"] for x in expert_stats),
        "trainable": sum(x["trainable"] for x in expert_stats),
        "non_trainable": sum(x["non_trainable"] for x in expert_stats),
    }
    wrapper_stat = _count_params(model)

    outside_expert = {
        "total": wrapper_stat["total"] - experts_sum["total"],
        "trainable": wrapper_stat["trainable"] - experts_sum["trainable"],
        "non_trainable": wrapper_stat["non_trainable"] - experts_sum["non_trainable"],
    }

    same_layout = all(
        st["total"] == single_stat["total"] and st["trainable"] == single_stat["trainable"]
        for st in expert_stats
    )

    log_func("=" * 80)
    log_func("MULTI-EXPERT PARAMETER SUMMARY")
    log_func("=" * 80)
    log_func(f"Number of Experts:       {num_experts}")
    log_func("-" * 80)
    log_func(
        f"Single Expert (expert_0): total={_format_params_lazy(single_stat['total'])}, "
        f"trainable={_format_params_lazy(single_stat['trainable'])}, "
        f"non_trainable={_format_params_lazy(single_stat['non_trainable'])}"
    )

    if same_layout:
        log_func(
            f"All Experts Sum:         {num_experts} x {_format_params_lazy(single_stat['total'])} "
            f"= {_format_params_lazy(experts_sum['total'])} "
            f"(trainable={_format_params_lazy(experts_sum['trainable'])})"
        )
    else:
        log_func(
            f"All Experts Sum:         total={_format_params_lazy(experts_sum['total'])}, "
            f"trainable={_format_params_lazy(experts_sum['trainable'])}, "
            f"non_trainable={_format_params_lazy(experts_sum['non_trainable'])}"
        )

    log_func(
        f"Wrapper model.parameters(): total={_format_params_lazy(wrapper_stat['total'])}, "
        f"trainable={_format_params_lazy(wrapper_stat['trainable'])}, "
        f"non_trainable={_format_params_lazy(wrapper_stat['non_trainable'])}"
    )

    if any(v != 0 for v in outside_expert.values()):
        log_func(
            f"Params outside experts:  total={_format_params_lazy(outside_expert['total'])}, "
            f"trainable={_format_params_lazy(outside_expert['trainable'])}, "
            f"non_trainable={_format_params_lazy(outside_expert['non_trainable'])}"
        )

    log_func("=" * 80)
    log_func("DETAILED BREAKDOWN OF SINGLE EXPERT (expert_0)")
    log_func("=" * 80)
    print_model_params_detailed(model.experts[0], logger=logger, max_depth=max_depth)


def _is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def _derive_rank_log_path(log_path: Optional[str], rank: int) -> Optional[str]:
    if log_path is None:
        return None
    p = Path(log_path)
    if rank == 0:
        return str(p)
    suffix = p.suffix if p.suffix else ".txt"
    return str(p.with_name(f"{p.stem}.rank{rank}{suffix}"))


def _destroy_process_group_safely():
    if _is_dist_ready():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _configure_debug_env(train_opt: Dict[str, Any]):
    if bool(train_opt.get("ddp_debug_detail", False)):
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

    if bool(train_opt.get("nccl_debug", False)):
        os.environ.setdefault("NCCL_DEBUG", str(train_opt.get("nccl_debug_level", "INFO")))

    if bool(train_opt.get("cuda_launch_blocking", False)):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if bool(train_opt.get("nccl_async_error_handling", True)):
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def _configure_runtime_perf(train_opt: Dict[str, Any]):
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
        f"[runtime] cudnn_benchmark={cudnn_benchmark}, allow_tf32={allow_tf32}, "
        f"float32_matmul_precision={matmul_precision or 'default'}"
    )


def _ddp_spawn_worker(
    rank: int,
    world_size: int,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    log_level: int,
    log_path: Optional[str],
    kwargs: Dict[str, Any]
):
    # 让被 mp.spawn SIGTERM 的进程也尽量 destroy pg，减少 NCCL warning
    def _term_handler(signum, frame):
        _destroy_process_group_safely()
        raise SystemExit(128 + int(signum))

    try:
        signal.signal(signal.SIGTERM, _term_handler)
        signal.signal(signal.SIGINT, _term_handler)
    except Exception:
        pass

    jdata = normalize(j_loader(INPUT))
    train_opt = jdata.get("train_options", {})

    _configure_debug_env(train_opt)

    backend = str(train_opt.get("ddp_backend", "nccl" if torch.cuda.is_available() else "gloo"))
    timeout_sec = int(train_opt.get("ddp_timeout_sec", 1800))

    os.environ.setdefault("MASTER_ADDR", str(train_opt.get("ddp_master_addr", "127.0.0.1")))
    os.environ.setdefault("MASTER_PORT", str(train_opt.get("ddp_master_port", "29501")))

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_sec),
    )

    try:
        _multi_train_impl(
            INPUT=INPUT,
            init_model=init_model,
            restart=restart,
            output=output,
            log_level=log_level,
            log_path=log_path,
            distributed_expert=True,
            rank=rank,
            world_size=world_size,
            **kwargs,
        )
    finally:
        _destroy_process_group_safely()


def _multi_train_impl(
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    log_level: int,
    log_path: Optional[str],
    distributed_expert: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
):
    run_opt: Dict[str, Any] = {
        "init_model": init_model,
        "restart": restart,
        "log_path": log_path,
        "log_level": log_level
    }

    if all((run_opt["init_model"], restart)):
        raise RuntimeError("--init-model and --restart should not be set at the same time")

    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        log_path = _derive_rank_log_path(log_path, rank) if distributed_expert else log_path
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
            "output": str(Path(output).absolute()),
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "log_path": str(Path(log_path).absolute())
        })
    else:
        if distributed_expert:
            log_path = _derive_rank_log_path(log_path, rank)

    set_log_handles(log_level, Path(log_path) if log_path else None)

    if sys.platform.startswith('win'):
        for handler in logging.root.handlers + logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.close()
                handler.stream = open(handler.baseFilename, handler.mode, encoding='utf-8')
            elif isinstance(handler, logging.StreamHandler):
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except Exception:
                    pass

    jdata = j_loader(INPUT)
    jdata = normalize(jdata)

    _configure_debug_env(jdata.get("train_options", {}))
    _configure_runtime_perf(jdata.get("train_options", {}))

    dbg = jdata.get("train_options", {})
    entry_tagger = _EntryTagger(
        enabled=bool(dbg.get("debug_tags", False)),
        cuda_mem=bool(dbg.get("debug_tag_cuda_mem", True)),
        cuda_sync=bool(dbg.get("debug_tag_cuda_sync", False)),
    )

    if distributed_expert:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed_expert=True requires CUDA.")
        jdata["common_options"]["device"] = f"cuda:{rank}"
        jdata["train_options"]["use_ddp"] = True
        jdata["train_options"]["ddp_world_size"] = world_size
        jdata["train_options"]["ddp_rank"] = rank

    with entry_tagger.tag("set_default_dtype"):
        torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    with entry_tagger.tag("merge_config_from_ckpt_or_restart"):
        if restart or init_model:
            f = restart if restart else init_model
            if f.split(".")[-1] == "json":
                assert not restart, "json model can not be used as restart! should be a checkpoint file"
            else:
                f = torch.load(f, map_location="cpu", weights_only=False)
                if jdata.get("model_options", None) is None:
                    jdata["model_options"] = f["config"]["model_options"]

                basis = f["config"]["common_options"]["basis"]
                if len(f["config"]["model_options"]) == 1 and f["config"]["model_options"].get("nnsk") is not None:
                    for asym, orb in jdata["common_options"]["basis"].items():
                        assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                        if orb != basis[asym]:
                            log.info(f"Initializing Orbital {orb} of Atom {asym} from {basis[asym]}")
                    for asym, orb in basis.items():
                        if asym not in jdata["common_options"]["basis"].keys():
                            jdata["common_options"]["basis"][asym] = orb
                else:
                    for asym, orb in jdata["common_options"]["basis"].items():
                        assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                        assert orb == basis[asym], f"Orbital {orb} of Atom {asym} not consistent with the model's basis."
                    jdata["common_options"]["basis"] = basis

                if restart:
                    if jdata.get("train_options", None) is not None:
                        for obj in MultiTrainer.object_keys:
                            if jdata["train_options"].get(obj) != f["config"]["train_options"].get(obj):
                                log.warning(f"{obj} in config file is not consistent with the checkpoint, using the one in checkpoint")
                                jdata["train_options"][obj] = f["config"]["train_options"][obj]
                    else:
                        jdata["train_options"] = f["config"]["train_options"]

                    if jdata.get("model_options", None) is None or jdata["model_options"] != f["config"]["model_options"]:
                        jdata["model_options"] = f["config"]["model_options"]
                else:
                    if jdata.get("train_options", None) is None:
                        jdata["train_options"] = f["config"]["train_options"]
                    if jdata.get("model_options") is None:
                        jdata["model_options"] = f["config"]["model_options"]
                    for k, v in jdata["model_options"].items():
                        if k not in f["config"]["model_options"]:
                            log.warning(f"The model options {k} is not defined in checkpoint, set to {v}.")
                        else:
                            deep_dict_difference(k, v, f["config"]["model_options"])
                del f
        else:
            j_must_have(jdata, "model_options")
            j_must_have(jdata, "train_options")

    if distributed_expert:
        jdata["common_options"]["device"] = f"cuda:{rank}"
        jdata["train_options"]["use_ddp"] = True
        jdata["train_options"]["ddp_world_size"] = world_size
        jdata["train_options"]["ddp_rank"] = rank

    cutoff_options = collect_cutoffs(jdata)

    with entry_tagger.tag("setup_seed"):
        setup_seed(seed=jdata["common_options"]["seed"])

    with entry_tagger.tag("build_dataset/train"):
        train_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["train"], **jdata["common_options"])

    validation_datasets = None
    if jdata["data_options"].get("validation"):
        with entry_tagger.tag("build_dataset/validation"):
            validation_datasets = build_dataset(
                **cutoff_options,
                **jdata["data_options"]["validation"],
                **jdata["common_options"]
            )

    reference_datasets = None
    if jdata["data_options"].get("reference"):
        with entry_tagger.tag("build_dataset/reference"):
            reference_datasets = build_dataset(
                **cutoff_options,
                **jdata["data_options"]["reference"],
                **jdata["common_options"]
            )

    jdata["common_options"]["overlap"] = False

    distance_ranges = jdata["train_options"].get(
        "distance_ranges",
        [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
    )

    parallel_multi = bool(
        jdata["train_options"].get(
            "parallel_multi",
            jdata["train_options"].get("parallel_forward", False)
        )
    )

    if distributed_expert:
        if len(distance_ranges) != world_size:
            raise ValueError(
                f"In distributed_expert mode, world_size must equal len(distance_ranges). "
                f"Got world_size={world_size}, len(distance_ranges)={len(distance_ranges)}"
            )
        if parallel_multi:
            log.warning("distributed_expert=True: force disable parallel_multi.")
        parallel_multi = False

    jdata["train_options"]["parallel_multi"] = parallel_multi

    jdata["train_options"]["log_single_model_compatible_loss"] = bool(
        jdata["train_options"].get("log_single_model_compatible_loss", True)
    )
    jdata["train_options"]["log_single_model_compatible_loss_mode"] = str(
        jdata["train_options"].get("log_single_model_compatible_loss_mode", "reduce")
    )

    log.info(f"[MultiTrainer][rank={rank}] distributed_expert = {distributed_expert}")
    log.info(f"[MultiTrainer][rank={rank}] parallel_multi = {parallel_multi}")
    log.info(f"[MultiTrainer][rank={rank}] distributed_rank0_prepare_batch = {jdata['train_options'].get('distributed_rank0_prepare_batch', False)}")
    log.info(f"[MultiTrainer][rank={rank}] train_num_workers = {jdata['train_options'].get('train_num_workers', jdata['train_options'].get('num_workers', 0))}")
    log.info(
        f"[MultiTrainer][rank={rank}] log_single_model_compatible_loss = "
        f"{jdata['train_options']['log_single_model_compatible_loss']}, "
        f"mode={jdata['train_options']['log_single_model_compatible_loss_mode']}"
    )

    if restart:
        with entry_tagger.tag("trainer/restart"):
            trainer = MultiTrainer.restart(
                checkpoint=restart,
                train_datasets=train_datasets,
                train_options=jdata["train_options"],
                common_options=jdata["common_options"],
                reference_datasets=reference_datasets,
                validation_datasets=validation_datasets,
                distributed_expert=distributed_expert,
                rank=rank,
                world_size=world_size,
            )
    else:
        checkpoint = init_model if init_model else None

        with entry_tagger.tag("build_model"):
            model = build_model(
                checkpoint=checkpoint,
                model_options=jdata["model_options"],
                common_options=jdata["common_options"],
                train_options=jdata["train_options"]
            )

        scale_type = jdata["model_options"]["prediction"].get('scale_type', "scale_w_back_grad")
        if scale_type == 'no_scale':
            log.info('Skip the E3statistics part, since the scale_type is no_scale')
        else:
            with entry_tagger.tag("dataset/E3statistics", device=torch.device(jdata["common_options"]["device"])):
                log.info(f'Start the E3statistics part, since the scale_type is {scale_type}')
                train_datasets.E3statistics(model=model)

        with entry_tagger.tag("trainer/init"):
            trainer = MultiTrainer(
                distance_ranges=distance_ranges,
                train_options=jdata["train_options"],
                common_options=jdata["common_options"],
                model=model,
                train_datasets=train_datasets,
                validation_datasets=validation_datasets,
                reference_datasets=reference_datasets,
                distributed_expert=distributed_expert,
                rank=rank,
                world_size=world_size,
            )

    with entry_tagger.tag("trainer/register_plugins"):
        log_field = ["train_loss", "train_loss_opt", "lr"]

        if validation_datasets:
            trainer.register_plugin(
                Validationer(
                    interval=[(jdata["train_options"]["validation_freq"], 'iteration'), (1, 'epoch')],
                    fast_mode=jdata["train_options"]["valid_fast"]
                )
            )
            log_field.append("validation_loss")

        avg_per_iter = chk_avg_per_iter(jdata)

        trainer.register_plugin(
            TrainLossMonitor(
                sliding_win_size=jdata["train_options"]["sliding_win_size"],
                avg_per_iter=avg_per_iter
            )
        )
        trainer.register_plugin(LearningRateMonitor())

        trainer.register_plugin(TrainOnsiteLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(TrainHoppingLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(TrainZLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(ExpertLoadCVMonitor(interval=[(1, 'iteration'), (1, 'epoch')]))
        trainer.register_plugin(ScalarFieldMonitor(stat_name="train_loss_opt", interval=[(1, 'iteration'), (1, 'epoch')]))

        for i in range(trainer.num_experts):
            trainer.register_plugin(ScalarFieldMonitor(stat_name=f"expert_{i}_onsite", interval=[(1, 'iteration'), (1, 'epoch')]))
            trainer.register_plugin(ScalarFieldMonitor(stat_name=f"expert_{i}_hopping", interval=[(1, 'iteration'), (1, 'epoch')]))
            trainer.register_plugin(ScalarFieldMonitor(stat_name=f"expert_{i}_lr", interval=[(1, 'iteration'), (1, 'epoch')]))

        log_field.extend(["mean_max_prob", "expert_load_cv", "train_onsite_loss", "train_hopping_loss"])

        if trainer.is_main_process:
            monitor_flag = jdata["train_options"].get("monitor_flag", False)
            if monitor_flag:
                trainer.register_plugin(DeepDoctorMonitor(output, verbose_freq=1))
                trainer.register_plugin(SO2ModuleMonitor(output))
                trainer.register_plugin(PreTPBlockMonitor(output))

            if jdata["train_options"].get("use_tensorboard"):
                tb_log_dir = os.path.join(output, "tensorboard_logs") if output else "./tensorboard_logs"
                trainer.register_plugin(
                    TensorBoardMonitor(
                        interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')],
                        log_dir=tb_log_dir
                    )
                )

            trainer.register_plugin(
                Logger(log_field, interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')])
            )

        for q in trainer.plugin_queues.values():
            heapq.heapify(q)

        if output and trainer.is_main_process:
            with open(os.path.join(output, "train_config.json"), "w") as fp:
                json.dump(jdata, fp, indent=4)

        if output and jdata["train_options"].get("save_freq"):
            trainer.register_plugin(
                Saver(interval=[(jdata["train_options"].get("save_freq"), 'iteration'), (1, 'epoch')]),
                checkpoint_path=run_opt["checkpoint_path"]
            )

    if trainer.is_main_process:
        print_multi_model_params_detailed(trainer.model, logger=log, max_depth=5)

    if distributed_expert and _is_dist_ready():
        dist.barrier()

    with entry_tagger.tag("trainer/run", device=torch.device(jdata["common_options"]["device"])):
        start_time = time.time()
        trainer.run(trainer.train_options["num_epoch"])
        end_time = time.time()

    if distributed_expert and _is_dist_ready():
        dist.barrier()

    if trainer.is_main_process:
        log.info("finished training")
        log.info(f"wall time: {(end_time - start_time):.3f} s")


def multi_train(
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    output: str,
    log_level: int,
    log_path: Optional[str],
    **kwargs
):
    jdata = normalize(j_loader(INPUT))
    train_opt = jdata.get("train_options", {})
    _configure_debug_env(train_opt)

    distance_ranges = train_opt.get(
        "distance_ranges",
        [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
    )
    use_ddp = bool(train_opt.get("use_ddp", False))

    if use_ddp and len(distance_ranges) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("use_ddp=True requires CUDA.")
        world_size = len(distance_ranges)
        n_gpu = torch.cuda.device_count()
        if n_gpu < world_size:
            raise RuntimeError(
                f"Not enough GPUs for distributed_expert mode: need {world_size}, but only {n_gpu} available."
            )

        mp.spawn(
            _ddp_spawn_worker,
            nprocs=world_size,
            args=(world_size, INPUT, init_model, restart, output, log_level, log_path, kwargs),
            join=True
        )
        return

    _multi_train_impl(
        INPUT=INPUT,
        init_model=init_model,
        restart=restart,
        output=output,
        log_level=log_level,
        log_path=log_path,
        distributed_expert=False,
        rank=0,
        world_size=1,
        **kwargs
    )


if __name__ == "__main__":
    import shutil
    if os.path.exists('output_dir_multi'):
        shutil.rmtree('output_dir_multi')

    multi_train(
        INPUT=r'general_debug_Al_10_cubic.json',
        output=r'output_dir_multi',
        log_level=2,
        log_path=r'log.txt',
        init_model=None,
        restart=None,
    )