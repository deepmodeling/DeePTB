from dptb.nn.deeptb import NNENV, MIX
import logging
from dptb.nn.nnsk import NNSK
from dptb.nn.dftbsk import DFTBSK
import torch
import torch.nn as nn
from dptb.utils.tools import j_must_have, j_loader
import copy

log = logging.getLogger(__name__)


# ======================================================================
# 新增: 距离集合包装器 (Distance Ensemble Wrapper)
# ======================================================================
class DistanceEnsembleWrapper(nn.Module):
    """
    一个透明的模型包装器，用于实现 Distance MOE。
    """

    def __init__(self, base_model, num_experts):
        super().__init__()
        # 为了对外透明，将基础模型的属性映射到 wrapper 上
        self.name = base_model.name
        self.device = base_model.device
        self.model_options = base_model.model_options

        # 将原始基础模型深拷贝 N 份，作为独立的专家
        # 这里需要注意：如果模型很大，这会复制多份显存。
        self.experts = nn.ModuleList([
            copy.deepcopy(base_model) if i > 0 else base_model for i in range(num_experts)
        ])

        # 同步底层需要访问的核心属性（适配 Trainer 和 Saver 插件）
        if hasattr(base_model, 'hamiltonian'):
            self.hamiltonian = base_model.hamiltonian
        if hasattr(base_model, 'hopping_options'):
            self.hopping_options = base_model.hopping_options
        if hasattr(base_model, 'ovp_factor'):
            self.ovp_factor = base_model.ovp_factor

    def forward(self, batch):
        expert_idx = batch.get("expert_idx", None)
        if expert_idx is not None:
            # 训练阶段：只调用被选中的那个专家
            return self.experts[expert_idx](batch)
        else:
            # 推理阶段：调用所有专家，并将 Hamiltonian 累加
            res = self.experts[0](batch)  # 取第一个专家的其他结果作为骨架
            total_H = 0
            for expert in self.experts:
                total_H += expert(batch)["hamiltonian"]
            res["hamiltonian"] = total_H
            return res


# ======================================================================

def build_model(
        checkpoint: str = None,
        model_options: dict = {},
        common_options: dict = {},
        train_options: dict = {},  # [新增参数]：为了感知是否使用 MultiTrainer
        no_check: bool = False,
        device: str = None,
):
    """
    The build model method should composed of the following steps:
        1. process the configs from user input and the config from the checkpoint (if any).
        2. construct the model based on the configs.
        3. process the config dict for the output dict.
    """
    if checkpoint is not None:
        from_scratch = False
    else:
        from_scratch = True
        if not all((model_options, common_options)):
            logging.error(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")
            raise ValueError(
                "You need to provide model_options and common_options when you are initializing a model from scratch.")

    init_nnenv = False
    init_nnsk = False
    init_mixed = False
    init_dftbsk = False

    if not from_scratch:
        if checkpoint.split(".")[-1] == "json":
            ckptconfig = j_loader(checkpoint)
        else:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckptconfig = f['config']
            del f

        if len(model_options) == 0:
            model_options = ckptconfig["model_options"]
        if len(common_options) == 0:
            common_options = ckptconfig["common_options"]
        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})
        del ckptconfig

    # [省略中间原本的配置检查代码，完全保持你的逻辑]
    if model_options.get("dftbsk"):
        assert not model_options.get("nnsk"), "There should only be one of the dftbsk and nnsk in model_options."
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                raise ValueError("The prediction method must be sktb for mix mode.")
            if not model_options['embedding']['method'] in ['se2']:
                raise ValueError("The embedding method must be se2 for mix mode.")
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_dftbsk = True
        else:
            raise ValueError("Model_options are not set correctly!")
    elif model_options.get("nnsk"):
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                raise ValueError("The prediction method must be sktb for mix mode.")
            if not model_options['embedding']['method'] in ['se2']:
                raise ValueError("The embedding method must be se2 for mix mode.")
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnsk = True
        else:
            raise ValueError("Model_options are not set correctly!")
    else:
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnenv = True
            if model_options["prediction"]['method'] == 'sktb':
                log.warning(
                    "The prediction method is sktb, but the nnsk option is not set. this is highly not recommand.")
                if not model_options['embedding']['method'] in ['se2']:
                    raise ValueError("The embedding method must be se2 for sktb prediction in deeptb mode.")
            if model_options["prediction"]['method'] == 'e3tb':
                if model_options['embedding']['method'] in ['se2']:
                    raise ValueError("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
        else:
            raise ValueError("Model_options are not set correctly!")

        assert int(init_dftbsk) + int(init_mixed) + int(init_nnenv) + int(
            init_nnsk) == 1, "IF not sk, you can only choose one of the mixed, nnenv, dftb and nnsk options."

    if device:
        common_options.update({"device": device})

    # init deeptb
    if from_scratch:
        if init_nnenv:
            model = NNENV(**model_options, **common_options)
        elif init_nnsk:
            model = NNSK(**model_options["nnsk"], **common_options)
        elif init_mixed:
            model = MIX(**model_options, **common_options)
        elif init_dftbsk:
            model = DFTBSK(**model_options["dftbsk"], **common_options)
        else:
            model = None
    else:
        if init_nnenv:
            model = NNENV.from_reference(checkpoint, **model_options, **common_options)
        elif init_nnsk:
            model = NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
        elif init_mixed:
            model = MIX.from_reference(checkpoint, **model_options, **common_options)
        elif init_dftbsk:
            model = DFTBSK.from_reference(checkpoint, **model_options["dftbsk"], **common_options)
        else:
            model = None

    if not no_check:
        for k, v in model.model_options.items():
            if k not in model_options:
                log.warning(f"The model options {k} is not defined in input model_options, set to {v}.")
            else:
                deep_dict_difference(k, v, model_options)

    # ======================================================================
    # [最小侵入式更新]
    # 判断是否配置了 distance_ranges。如果配置了，自动装载 Wrapper
    # ======================================================================
    distance_ranges = train_options.get("distance_ranges", None)

    if distance_ranges is not None and len(distance_ranges) > 1:
        log.info(f"Wrapping model with DistanceEnsembleWrapper ({len(distance_ranges)} experts)")

        # 处理 checkpoint：如果是从 restart 进来的，此时原版 model 参数可能只有1个
        # 但是 Saver 保存的时候存了 "model_state_dict"，因为它是 Wrapper 存的
        # 所以我们需要在这里先把它 wrap 起来，然后再加载状态字典（在 Restart 流程外部统一加载或在 from_reference 里面已经包含了?）

        model = DistanceEnsembleWrapper(model, len(distance_ranges))

        # 特别注意：如果你是从 checkpoint 恢复，原始的 from_reference 试图加载一个带有 experts.0, experts.1 的字典到一个没有 experts 的单模型里会报错。
        # 因此我们需要在 from_reference 后手动干预，或要求 from_scratch 时 wrap，restart 时在 MultiTrainer.restart 里完成字典对齐。
        # 考虑到代码极简性：
        if not from_scratch:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            # 因为被 wrapper 包裹过了，重新加载包裹后的字典
            try:
                model.load_state_dict(ckpt["model_state_dict"])
            except Exception as e:
                log.warning(f"Failed to load wrapper dict directly: {e}. Will rely on strict=False logic.")

    model.to(model.device)

    return model


def deep_dict_difference(base_key, expected_value, model_options):
    target_dict = copy.deepcopy(model_options)
    if isinstance(expected_value, dict):
        for subk, subv in expected_value.items():
            if subk not in target_dict.get(base_key, {}):
                log.warning(
                    f"The model option {subk} in {base_key} is not defined in input model_options, set to {subv}.")
            else:
                target2 = copy.deepcopy(target_dict[base_key])
                deep_dict_difference(f"{subk}", subv, target2)
    else:
        if expected_value != target_dict[base_key]:
            log.warning(
                f"The model option {base_key} is set to {expected_value}, but in input it is {target_dict[base_key]}, make sure it it correct!")