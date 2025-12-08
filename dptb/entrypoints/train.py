from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model
from dptb.data.build import build_dataset
from dptb.plugins.monitor import TrainLossMonitor, LearningRateMonitor, Validationer, TensorBoardMonitor, DeepDoctorMonitor, SO2ModuleMonitor, PreTPBlockMonitor
from dptb.plugins.train_logger import Logger
from dptb.utils.argcheck import normalize, collect_cutoffs, chk_avg_per_iter
from dptb.plugins.saver import Saver
from typing import Tuple, Dict, List, Optional, Any
from dptb.utils.tools import j_loader, setup_seed, j_must_have
from dptb.utils.constants import dtype_dict
from dptb.utils.loggers import set_log_handles
import heapq
import logging
import torch
import random
import numpy as np
from pathlib import Path
import json
import os
import time
import copy
import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ["train"]

log = logging.getLogger(__name__)

from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


def format_params(num_params: int) -> str:
    """格式化参数数量，显示不同单位"""
    if num_params >= 1e9:
        return f"{num_params:,} ({num_params / 1e9:.4f}B)"
    elif num_params >= 1e6:
        return f"{num_params:,} ({num_params / 1e6:.3f}M)"
    elif num_params >= 1e3:
        return f"{num_params:,} ({num_params / 1e3:.2f}K)"
    else:
        return f"{num_params:,}"


def get_module_params(module: nn.Module) -> Tuple[int, int]:
    """获取模块的可训练参数和总参数数量"""
    trainable = 0
    total = 0

    # 只计算直接属于该模块的参数，不包括子模块
    for name, param in module.named_parameters(recurse=False):
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    return trainable, total


def simplify_irreps_str(irreps_obj) -> str:
    """
    简化irreps字符串，合并同类项并按l排序

    Args:
        irreps_obj: o3.Irreps对象或可转换为Irreps的对象

    Returns:
        简化后的字符串表示
    """
    try:
        # 尝试导入e3nn，如果不存在则返回原始字符串
        from e3nn import o3

        # 确保是Irreps对象
        if not isinstance(irreps_obj, o3.Irreps):
            irreps_obj = o3.Irreps(irreps_obj)

        # 合并相同的不可约表示
        merged_dict = {}
        for mul, ir in irreps_obj:
            key = (ir.l, ir.p)  # 使用(l, parity)作为键
            if key in merged_dict:
                merged_dict[key] += mul
            else:
                merged_dict[key] = mul

        # 按l值排序，如果l相同则按奇偶性排序（偶数在前）
        simplified_parts = []
        for (l, p) in sorted(merged_dict.keys(), key=lambda x: (x[0], -x[1])):
            mul = merged_dict[(l, p)]
            simplified_parts.append(f"{mul}x{l}{'e' if p == 1 else 'o'}")

        return '+'.join(simplified_parts)

    except (ImportError, Exception):
        # 如果e3nn不可用或转换失败，返回原始字符串
        return str(irreps_obj)


def get_module_irreps(module: nn.Module) -> Dict[str, str]:
    """获取模块的irreps信息（简化版本）"""
    irreps_info = {}

    # 检查 irreps_in
    if hasattr(module, 'irreps_in'):
        irreps_info['irreps_in'] = simplify_irreps_str(module.irreps_in)

    # 检查 irreps_out
    if hasattr(module, 'irreps_out'):
        irreps_info['irreps_out'] = simplify_irreps_str(module.irreps_out)

    return irreps_info


def analyze_model_params(model: nn.Module, max_depth: int = 3) -> Dict[str, Any]:
    """
    分析模型参数的详细信息

    Args:
        model: PyTorch模型
        max_depth: 最大递归深度

    Returns:
        包含参数统计信息的字典
    """
    stats = OrderedDict()

    def recursive_analyze(module: nn.Module, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        # 获取当前模块的参数
        trainable, total = get_module_params(module)

        # 计算包括子模块的总参数
        trainable_with_children = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_with_children = sum(p.numel() for p in module.parameters())

        # 获取irreps信息
        irreps_info = get_module_irreps(module)

        # 存储当前模块信息
        if prefix or depth == 0:  # 根模块或有名称的模块
            module_name = prefix if prefix else "Model"
            stats[module_name] = {
                'trainable': trainable_with_children,
                'total': total_with_children,
                'own_trainable': trainable,
                'own_total': total,
                'depth': depth,
                'type': module.__class__.__name__,
                'irreps': irreps_info  # 新增irreps信息
            }

        # 递归分析子模块
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            recursive_analyze(child, child_prefix, depth + 1)

    recursive_analyze(model)
    return stats


def print_model_params_detailed(model: nn.Module, logger=None, max_depth: int = 3):
    """
    详细打印模型参数信息

    Args:
        model: PyTorch模型
        logger: 日志对象（可选）
        max_depth: 显示的最大层级深度
    """
    # 如果logger为None，使用print
    log_func = logger.info if logger else print

    # 获取总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # 打印总体统计
    log_func("=" * 80)
    log_func("MODEL PARAMETERS SUMMARY")
    log_func("=" * 80)

    log_func(f"Total Parameters:        {format_params(total_params)}")
    log_func(f"Trainable Parameters:    {format_params(trainable_params)}")
    log_func(f"Non-trainable Parameters: {format_params(non_trainable_params)}")
    log_func(f"Trainable Ratio:         {trainable_params / total_params * 100:.2f}%")
    log_func("-" * 80)

    # 获取详细统计
    stats = analyze_model_params(model, max_depth)

    # 打印详细信息
    log_func("DETAILED BREAKDOWN BY MODULE:")
    log_func("-" * 80)

    # 打印表头
    header = f"{'Module':<50} {'Type':<20} {'Trainable':<20} {'Total':<20}"
    log_func(header)
    log_func("-" * 80)

    prev_parents = {}  # 用于追踪不同层级的父模块

    for idx, (module_name, info) in enumerate(stats.items()):
        indent = "  " * info['depth']
        name = indent + module_name.split('.')[-1] if '.' in module_name else module_name

        # 截断过长的名称
        if len(name) > 48:
            name = name[:45] + "..."

        # 根据深度添加不同样式的分隔线
        parts = module_name.split('.')
        should_print_separator = False
        separator_char = None

        # depth=1: 主模块之间 (如 init_layer -> layers.0, layers.0 -> layers.1)
        if info['depth'] == 1:
            current_key = parts[0] if len(parts) > 0 else module_name
            if 1 in prev_parents and prev_parents[1] != current_key:
                should_print_separator = True
                separator_char = "═"
            prev_parents[1] = current_key

        # depth=2: 二级模块之间 (如 edge_update -> node_update)
        if info['depth'] == 2:
            current_key = '.'.join(parts[:2]) if len(parts) >= 2 else module_name
            if 2 in prev_parents and prev_parents[2] != current_key:
                should_print_separator = True
                separator_char = "─"
            prev_parents[2] = current_key

        # depth=3: 三级模块之间
        if info['depth'] == 3:
            current_key = '.'.join(parts[:3]) if len(parts) >= 3 else module_name
            if 3 in prev_parents and prev_parents[3] != current_key:
                should_print_separator = True
                separator_char = "·"
            prev_parents[3] = current_key

        # 打印分隔线（如果需要）
        if should_print_separator and separator_char and idx > 0:
            log_func(separator_char * 80)

        # 格式化输出
        if info['own_trainable'] > 0 or info['own_total'] > 0:
            # 如果模块有自己的参数，显示详细信息
            row = f"{name:<50} {info['type']:<20} "
            row += f"{format_params(info['trainable']):<20} "
            row += f"{format_params(info['total']):<20}"
            log_func(row)

            # 打印irreps信息（如果存在）
            if info['irreps']:
                irreps_indent = indent + "  "
                if 'irreps_in' in info['irreps']:
                    log_func(f"{irreps_indent}↳ irreps_in:  {info['irreps']['irreps_in']}")
                if 'irreps_out' in info['irreps']:
                    log_func(f"{irreps_indent}↳ irreps_out: {info['irreps']['irreps_out']}")

            # 如果模块自己的参数和总参数不同（即有子模块），显示自己的参数
            if info['own_total'] != info['total'] and info['own_total'] > 0:
                own_row = f"{indent + '  (own)':<50} {'':<20} "
                own_row += f"{format_params(info['own_trainable']):<20} "
                own_row += f"{format_params(info['own_total']):<20}"
                log_func(own_row)
        else:
            # 容器模块（没有自己的参数）
            row = f"{name:<50} {info['type']:<20} "
            row += f"{format_params(info['trainable']):<20} "
            row += f"{format_params(info['total']):<20}"
            log_func(row)

            # 打印irreps信息（如果存在）
            if info['irreps']:
                irreps_indent = indent + "  "
                if 'irreps_in' in info['irreps']:
                    log_func(f"{irreps_indent}↳ irreps_in:  {info['irreps']['irreps_in']}")
                if 'irreps_out' in info['irreps']:
                    log_func(f"{irreps_indent}↳ irreps_out: {info['irreps']['irreps_out']}")

    log_func("=" * 80)

    # 打印参数形状统计（可选）
    log_func("\nPARAMETER SHAPES SUMMARY:")
    log_func("-" * 80)

    param_shapes = {}
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        if shape not in param_shapes:
            param_shapes[shape] = {'count': 0, 'total_params': 0, 'names': []}
        param_shapes[shape]['count'] += 1
        param_shapes[shape]['total_params'] += param.numel()
        if len(param_shapes[shape]['names']) < 3:  # 只保存前3个例子
            param_shapes[shape]['names'].append(name.split('.')[-1])

    # 按参数数量排序
    sorted_shapes = sorted(param_shapes.items(), key=lambda x: x[1]['total_params'], reverse=True)

    log_func(f"{'Shape':<30} {'Count':<10} {'Total Params':<20} {'Examples':<30}")
    log_func("-" * 80)

    for shape, info in sorted_shapes[:10]:  # 只显示前10个最大的
        shape_str = str(shape)
        if len(shape_str) > 28:
            shape_str = shape_str[:25] + "..."
        examples = ', '.join(info['names'][:2])
        if len(examples) > 28:
            examples = examples[:25] + "..."

        log_func(f"{shape_str:<30} {info['count']:<10} {format_params(info['total_params']):<20} {examples:<30}")

    log_func("=" * 80)


# 使用示例
def use_example(trainer, log):
    """
    使用示例，替换原有的简单打印代码

    原代码:
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    log.info(
        f"Model parameters: {total_params:,},  |    "
        f"(About {total_params / 1e3:.2f}K)  |    "
        f"(About {total_params / 1e6:.3f}M)  |    "
        f"(About {total_params / 1e9:.4f}B)"
    )
    """

    # 新的详细打印
    print_model_params_detailed(trainer.model, logger=log, max_depth=3)

    # 如果只想要简单的汇总信息
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    log.info(
        f"Quick Summary - Trainable parameters: {total_params:,}  |  "
        f"{total_params / 1e3:.2f}K  |  "
        f"{total_params / 1e6:.3f}M  |  "
        f"{total_params / 1e9:.4f}B"
    )

def train(
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


    '''
        -1- set up input and output directories
        -2- parse configuration file and start training
            
    output directories has following structure:
        - ./output/
            - checkpoint/
                - latest_dptb.pth
                - best_dptb.pth
                - latest_nnsk.pth
                ...
            - log/
                - log.log
            - config.json
    '''
    # init all paths
    # if init_model, restart or init_frez, findout the input configure file

    # setup INPUT path

    if all((run_opt["init_model"], restart)):
        raise RuntimeError(
            "--init-model and --restart should not be set at the same time"
        )

    # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        checkpoint_path = os.path.join(str(output), "checkpoint")
        Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

    # parse training configuration, if INPUT is None and restart or init model is True, we can load the configure of
    # checkpoint
        run_opt.update({
            "output": str(Path(output).absolute()),
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "log_path": str(Path(log_path).absolute())
        })

    set_log_handles(log_level, Path(log_path) if log_path else None)
    # parse the config. Since if use init, config file may not equals to current

    jdata = j_loader(INPUT)
    jdata = normalize(jdata)
    # update basis if init_model or restart
    # update jdata
    # this is not necessary, because if we init model from checkpoint, the build_model will load the model_options from checkpoints if not provided
    # since here we want to output jdata as a config file to inform the user what model options are used, we need to update the jdata

    torch.set_default_dtype(getattr(torch, jdata["common_options"]["dtype"]))

    if restart or init_model:

        f = restart if restart else init_model

        if f.split(".")[-1] == "json":
            assert not restart, "json model can not be used as restart! should be a checkpoint file"
        else:
            f = torch.load(f, map_location="cpu", weights_only=False)

            if jdata.get("model_options", None) is None:
                jdata["model_options"] = f["config"]["model_options"]

            # update basis
            basis = f["config"]["common_options"]["basis"]
            # nnsk
            if len(f["config"]["model_options"])==1 and f["config"]["model_options"].get("nnsk") != None:
                for asym, orb in jdata["common_options"]["basis"].items():
                    assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                    if orb != basis[asym]:
                        log.info(f"Initializing Orbital {orb} of Atom {asym} from {basis[asym]}")
                # we have the orbitals in jdata basis correct, now we need to make sure all atom in basis are also contained in jdata basis
                for asym, orb in basis.items():
                    if asym not in jdata["common_options"]["basis"].keys():
                        jdata["common_options"]["basis"][asym] = orb # add the atomtype in the checkpoint but not in the jdata basis, because it will be used to build the orbital mapper for dataset
            else: # not nnsk
                for asym, orb in jdata["common_options"]["basis"].items():
                    assert asym in basis.keys(), f"Atom {asym} not found in model's basis"
                    assert orb == basis[asym], f"Orbital {orb} of Atom {asym} not consistent with the model's basis, which is only allowed in nnsk training"

                jdata["common_options"]["basis"] = basis

            # update model options and train_options
            if restart:
                #
                if jdata.get("train_options", None) is not None:
                    for obj in Trainer.object_keys:
                        if jdata["train_options"].get(obj) != f["config"]["train_options"].get(obj):
                            log.warning(f"{obj} in config file is not consistent with the checkpoint, using the one in checkpoint")
                            jdata["train_options"][obj] = f["config"]["train_options"][obj]
                else:
                    jdata["train_options"] = f["config"]["train_options"] # restart can be preceeded without train_options

                if jdata.get("model_options", None) is None or jdata["model_options"] != f["config"]["model_options"]:
                    log.warning("model_options in config file is not consistent with the checkpoint, using the one in checkpoint")
                    jdata["model_options"] = f["config"]["model_options"] # restart does not allow to change model options
            else:
                # init model mode, allow model_options change (Would it cause some error later if the param mismatch?)
                if jdata.get("train_options", None) is None:
                    jdata["train_options"] = f["config"]["train_options"]
                if jdata.get("model_options") is None:
                    jdata["model_options"] = f["config"]["model_options"]

                ## add some warning !
                for k, v in jdata["model_options"].items():
                    if k not in f["config"]["model_options"]:
                        log.warning(f"The model options {k} is not defined in checkpoint, set to {v}.")
                    else:
                        deep_dict_difference(k, v, f["config"]["model_options"])
            del f
    else:
        j_must_have(jdata, "model_options")
        j_must_have(jdata, "train_options")

    cutoff_options =collect_cutoffs(jdata)
    # setup seed
    setup_seed(seed=jdata["common_options"]["seed"])

    # with open(os.path.join(output, "train_config.json"), "w") as fp:
    #     json.dump(jdata, fp, indent=4)

    # build dataset
    train_datasets = build_dataset(**cutoff_options,**jdata["data_options"]["train"], **jdata["common_options"])
    if jdata["data_options"].get("validation"):
        validation_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["validation"], **jdata["common_options"])
    else:
        validation_datasets = None
    if jdata["data_options"].get("reference"):
        reference_datasets = build_dataset(**cutoff_options, **jdata["data_options"]["reference"], **jdata["common_options"])
    else:
        reference_datasets = None

    if restart:
        trainer = Trainer.restart(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            checkpoint=restart,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )
    else:
        # include the init model and from scratch
        # build model will handle the init model cases where the model options provided is not equals to the ones in checkpoint.
        checkpoint = init_model if init_model else None
        model = build_model(checkpoint=checkpoint, model_options=jdata["model_options"], common_options=jdata["common_options"])
        scale_type = jdata["model_options"]["prediction"].get('scale_type', "scale_w_back_grad")
        if scale_type == 'no_scale':
            log.info('Skip the E3statistics part, since the scale_type is no_scale')
        else:
            log.info(f'Start the E3statistics part, since the scale_type is {scale_type}')
            train_datasets.E3statistics(model=model)
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            reference_datasets=reference_datasets,
        )

    # register the plugin in trainer, to tract training info
    log_field = ["train_loss", "lr"]
    if validation_datasets:
        trainer.register_plugin(Validationer(interval=[(jdata["train_options"]["validation_freq"], 'iteration'), (1, 'epoch')], fast_mode=jdata["train_options"]["valid_fast"]))

        log_field.append("validation_loss")
    avg_per_iter = chk_avg_per_iter(jdata)
    trainer.register_plugin(TrainLossMonitor(sliding_win_size=jdata["train_options"]["sliding_win_size"], avg_per_iter=avg_per_iter)) # by default, avg_per_iter is false, will not be activated.
    trainer.register_plugin(LearningRateMonitor())


    current_bs = jdata["train_options"]["batch_size"]
    grad_log_file = os.path.join(output, f"grad_trace_bs{current_bs}.csv")

    # # # 注册 Monitor
    # # # verbose_freq 控制打印频率，但 CSV 是实时记录的
    # doctor = DeepDoctorMonitor(output, verbose_freq=10)
    # trainer.register_plugin(doctor)
    # so2_monitor = SO2ModuleMonitor(output)
    # trainer.register_plugin(so2_monitor)
    # pre_so2_monitor = PreTPBlockMonitor(output)
    # trainer.register_plugin(pre_so2_monitor)

    if jdata["train_options"]["use_tensorboard"]:
        assert jdata["train_options"]["display_freq"] >= jdata["train_options"]["validation_freq"], 'The display frequency must be greater than the validation frequency.'
        trainer.register_plugin(TensorBoardMonitor(interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))
    trainer.register_plugin(Logger(log_field,
        interval=[(jdata["train_options"]["display_freq"], 'iteration'), (1, 'epoch')]))

    for q in trainer.plugin_queues.values():
        heapq.heapify(q)

    if output:
        # output training configurations:
        with open(os.path.join(output, "train_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        trainer.register_plugin(Saver(
            # interval=[(jdata["train_options"].get("save_freq"), 'epoch'), (1, 'iteration')] if jdata["train_options"].get(
            #    "save_freq") else None))
            interval=[(jdata["train_options"].get("save_freq"), 'iteration'),  (1, 'epoch')] if jdata["train_options"].get(
                "save_freq") else None), checkpoint_path=checkpoint_path)
        # add a plugin to save the training parameters of the model, with model_output as given path

    print_model_params_detailed(trainer.model, logger=log, max_depth=5)
    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    log.info(
        f"Model parameters: {total_params:,},  |    "
                f"(About {total_params / 1e3:.2f}K)  |    "
                f"(About {total_params / 1e6:.3f}M)  |    "
                f"(About {total_params / 1e9:.4f}B)"
    )
    # =======================================================

    start_time = time.time()

    trainer.run(trainer.train_options["num_epoch"])

    end_time = time.time()
    log.info("finished training")
    log.info(f"wall time: {(end_time - start_time):.3f} s")


def deep_dict_difference(base_key, expected_value, model_options):
    """
    递归地记录嵌套字典中的选项差异。

    :param base_key: 基础键名，用于构建警告消息的前缀。
    :param expected_value: 期望的值，可能是字典或非字典类型。
    :param model_options: 用于比较的模型选项字典。
    """
    target_dict= copy.deepcopy(model_options) # 防止修改原始字典
    if isinstance(expected_value, dict):
        for subk, subv in expected_value.items():

            if  not isinstance(target_dict.get(base_key, {}),dict):
                log.warning(f"The model option {subk} in {base_key} is not defined in  checkpoint, set to {subv}.")

            elif subk not in target_dict.get(base_key, {}):
                log.warning(f"The model option {subk} in {base_key} is not defined in  checkpoint, set to {subv}.")
            else:
                target2 = copy.deepcopy(target_dict[base_key])
                deep_dict_difference(f"{subk}", subv, target2)
    else:
        if expected_value != target_dict[base_key]:
            log.warning(f"The model option {base_key} is set to {expected_value}, but in checkpoint it is {target_dict[base_key]}, make sure it it correct!")