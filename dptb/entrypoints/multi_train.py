import os
import json
import time
import heapq
import logging
import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional

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
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def print_multi_model_params_detailed(model: nn.Module, logger=None, max_depth: int = 5):
    """
    多专家参数统计：
    1) 单个 expert（expert_0）
    2) 所有 experts 之和
    3) 整个 wrapper 的 model.parameters() 总和
    4) 如果有 experts 外的共享参数，也单独打印
    """
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
        max_show = min(num_experts, 8)
        for i in range(max_show):
            st = expert_stats[i]
            log_func(
                f"  expert_{i}: total={_format_params_lazy(st['total'])}, "
                f"trainable={_format_params_lazy(st['trainable'])}, "
                f"non_trainable={_format_params_lazy(st['non_trainable'])}"
            )
        if num_experts > max_show:
            log_func(f"  ... ({num_experts - max_show} more experts omitted)")

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


def multi_train(
        INPUT: str,
        init_model: Optional[str],
        restart: Optional[str],
        output: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    run_opt = {
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
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
            "output": str(Path(output).absolute()),
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "log_path": str(Path(log_path).absolute())
        })

    set_log_handles(log_level, Path(log_path) if log_path else None)

    # Windows 下日志 / 终端强制 UTF-8
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
    torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    # 配置合并逻辑
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
                            log.warning(
                                f"{obj} in config file is not consistent with the checkpoint, using the one in checkpoint")
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

    cutoff_options = collect_cutoffs(jdata)
    setup_seed(seed=jdata["common_options"]["seed"])

    train_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["train"], **jdata["common_options"])

    validation_datasets = None
    if jdata["data_options"].get("validation"):
        validation_datasets = build_dataset(
            **cutoff_options,
            **jdata["data_options"]["validation"],
            **jdata["common_options"]
        )

    reference_datasets = None
    if jdata["data_options"].get("reference"):
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
    jdata["train_options"]["parallel_multi"] = parallel_multi

    # 默认开启：日志里的 train_loss 采用“单模型兼容口径”
    jdata["train_options"]["log_single_model_compatible_loss"] = bool(
        jdata["train_options"].get("log_single_model_compatible_loss", True)
    )

    log.info(f"[MultiTrainer] parallel_multi = {parallel_multi}")
    log.info(
        f"[MultiTrainer] log_single_model_compatible_loss = "
        f"{jdata['train_options']['log_single_model_compatible_loss']}"
    )

    if restart:
        trainer = MultiTrainer.restart(
            checkpoint=restart,
            train_datasets=train_datasets,
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )
    else:
        checkpoint = init_model if init_model else None
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
            log.info(f'Start the E3statistics part, since the scale_type is {scale_type}')
            train_datasets.E3statistics(model=model)

        trainer = MultiTrainer(
            distance_ranges=distance_ranges,
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model=model,
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            reference_datasets=reference_datasets,
        )

    log_field = ["train_loss", "lr"]

    if validation_datasets:
        trainer.register_plugin(
            Validationer(
                interval=[(jdata["train_options"]["validation_freq"], 'iteration'), (1, 'epoch')],
                fast_mode=jdata["train_options"]["valid_fast"]
            )
        )
        log_field.append("validation_loss")

    avg_per_iter = chk_avg_per_iter(jdata)

    # train_loss：单模型可比口径
    trainer.register_plugin(
        TrainLossMonitor(
            sliding_win_size=jdata["train_options"]["sliding_win_size"],
            avg_per_iter=avg_per_iter
        )
    )
    trainer.register_plugin(LearningRateMonitor())

    # 推荐每 iteration 更新，保证 epoch_mean 和 TensorBoard 曲线都完整
    trainer.register_plugin(
        TrainOnsiteLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')])
    )
    trainer.register_plugin(
        TrainHoppingLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')])
    )
    trainer.register_plugin(
        TrainZLossMonitor(interval=[(1, 'iteration'), (1, 'epoch')])
    )
    trainer.register_plugin(
        ExpertLoadCVMonitor(interval=[(1, 'iteration'), (1, 'epoch')])
    )

    # 真实优化目标：Σ expert_loss，仅调试用，不建议直接和旧单模型汇报对比
    trainer.register_plugin(
        ScalarFieldMonitor(
            stat_name="train_loss_opt",
            interval=[(1, 'iteration'), (1, 'epoch')]
        )
    )

    # 单专家 loss 监控
    for i in range(trainer.num_experts):
        trainer.register_plugin(
            ScalarFieldMonitor(
                stat_name=f"expert_{i}_onsite",
                interval=[(1, 'iteration'), (1, 'epoch')]
            )
        )
        trainer.register_plugin(
            ScalarFieldMonitor(
                stat_name=f"expert_{i}_hopping",
                interval=[(1, 'iteration'), (1, 'epoch')]
            )
        )

    log_field.extend([
        "mean_max_prob",
        "expert_load_cv",
        "train_onsite_loss",
        "train_hopping_loss"
    ])

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

    if output:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        if jdata["train_options"].get("save_freq"):
            trainer.register_plugin(
                Saver(interval=[(jdata["train_options"].get("save_freq"), 'iteration'), (1, 'epoch')]),
                checkpoint_path=run_opt["checkpoint_path"]
            )

    print_multi_model_params_detailed(trainer.model, logger=log, max_depth=5)

    start_time = time.time()
    trainer.run(trainer.train_options["num_epoch"])
    end_time = time.time()

    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


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