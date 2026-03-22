from dptb.nn.deeptb import NNENV, MIX
import logging
from dptb.nn.nnsk import NNSK
from dptb.nn.dftbsk import DFTBSK
import torch
import torch.nn as nn
from dptb.utils.tools import j_loader
from dptb.data import AtomicDataDict
import copy
from dptb.data.AtomicDataDict import with_edge_vectors

log = logging.getLogger(__name__)


# ======================================================================
# Distance Ensemble Wrapper
# ======================================================================
class DistanceEnsembleWrapper(nn.Module):
    """
    距离多专家包装器

    训练 / 验证阶段:
        - 由 MultiTrainer 驱动，batch 中自带 expert_idx 和提前算好的 mask
        - Wrapper 仅做无开销路由，直接分发给对应 expert

    推理 / 预测阶段 (脱离 Trainer):
        - batch 中不带 expert_idx
        - 自动运行所有 expert
        - 自动计算距离 mask，并对 edge 对齐的输出做无缝拼接 (stitching)
    """

    def __init__(self, experts, distance_ranges):
        super().__init__()

        assert len(experts) == len(distance_ranges), \
            f"len(experts) != len(distance_ranges): {len(experts)} vs {len(distance_ranges)}"

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.experts = nn.ModuleList(experts)

        base_model = self.experts[0]

        # 常用基础属性保留一份，便于外部逻辑直接读取
        self.name = getattr(base_model, "name", "distance_ensemble")
        self.device = getattr(base_model, "device", torch.device("cpu"))
        self.dtype = getattr(base_model, "dtype", torch.float32)
        self.model_options = copy.deepcopy(getattr(base_model, "model_options", {}))

        self._excluded_stitch_keys = self._build_excluded_stitch_keys()

    def __getattr__(self, name):
        """
        若 wrapper 自己没有某属性，则自动代理到 experts[0]。
        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            modules = object.__getattribute__(self, "_modules")
            experts = modules.get("experts", None) if modules is not None else None
            if experts is not None and len(experts) > 0:
                base_model = experts[0]
                if hasattr(base_model, name):
                    return getattr(base_model, name)
            raise e

    def _build_excluded_stitch_keys(self):
        keys = {
            "expert_idx",
            "expert_edge_mask",
            "expert_node_mask",
            "__slices__",
            "__cumsum__",
            "__cat_dims__",
            "__num_nodes_list__",
            "__data_class__",
            "edge_vec",
            "edge_index",
            "edge_type",
            "atom_types",
            "batch",
            "ptr",
            "cell",
            "pbc",
            "pos",
            "positions",
            "edge_lengths"
        }

        for const_name in [
            "ATOM_TYPE_KEY",
            "EDGE_TYPE_KEY",
            "EDGE_INDEX_KEY",
            "EDGE_VEC_KEY",
            "POSITIONS_KEY",
            "BATCH_KEY",
            "CELL_KEY",
            "PBC_KEY",
        ]:
            v = getattr(AtomicDataDict, const_name, None)
            if v is not None:
                keys.add(v)

        return keys

    def _build_edge_mask(self, batch, expert_idx):
        """
        推理阶段专属逻辑：自动根据距离生成 Mask
        优先复用数据管道的 edge_lengths，避免重复计算 torch.norm
        """
        if "edge_lengths" in batch:
            dist = batch["edge_lengths"]
        else:
            batch = with_edge_vectors(batch, with_lengths=True)
            dist = batch["edge_lengths"]

        d_min, d_max = self.distance_ranges[expert_idx]

        # 逻辑必须与 MultiTrainer 完全一致：最后一个专家“包圆”所有超距边
        if expert_idx == self.num_experts - 1:
            mask = (dist >= d_min)
        else:
            mask = (dist >= d_min) & (dist < d_max)

        return mask

    def _stitch_edge_aligned_outputs(self, res, res_i, mask):
        """自动拼接所有“第一维 == num_edges”的 tensor 输出"""
        num_edges = mask.shape[0]

        for key, src in res_i.items():
            if key in self._excluded_stitch_keys:
                continue
            if key not in res:
                continue

            dst = res[key]

            if not torch.is_tensor(src) or not torch.is_tensor(dst):
                continue
            if src.ndim == 0 or dst.ndim == 0:
                continue
            if src.shape != dst.shape:
                continue
            if src.shape[0] != num_edges:
                continue

            # 替换属于当前专家负责的边的数据
            dst[mask] = src[mask]

    def forward(self, batch):
        expert_idx = batch.get("expert_idx", None)

        # =========================
        # 训练 / 验证阶段 (走 Trainer)
        # =========================
        # Trainer 已经算好了 Mask，这里仅做零开销路由，无冗余计算
        if expert_idx is not None:
            return self.experts[int(expert_idx)](batch)

        # =========================
        # 推理阶段 (纯正向传播)
        # =========================
        # expert 0 负责 onsite + 第一段 hopping，先作为底层骨架
        res = self.experts[0](batch)

        # 后续 expert 用自己的距离区间覆盖 edge 对齐输出
        for i in range(1, self.num_experts):
            mask = self._build_edge_mask(batch, i)

            # 如果当前子图/结构中没有属于该距离区间的边，直接跳过，节省算力
            if not mask.any():
                continue

            res_i = self.experts[i](batch)
            self._stitch_edge_aligned_outputs(res, res_i, mask)

        return res


# ======================================================================
# Helper Functions
# ======================================================================
def _infer_num_xgrid_from_state_dict(state_dict: dict):
    if state_dict is None:
        return None
    for k, v in state_dict.items():
        if k.endswith("distance_param") and torch.is_tensor(v) and v.ndim >= 1:
            return int(v.shape[0])
    return None


def _construct_single_model(
        init_nnenv: bool,
        init_nnsk: bool,
        init_mixed: bool,
        init_dftbsk: bool,
        model_options: dict,
        common_options: dict,
        ref_state_dict: dict = None,
):
    extra_kwargs = {}

    num_xgrid = _infer_num_xgrid_from_state_dict(ref_state_dict)
    if num_xgrid is not None:
        if init_dftbsk:
            extra_kwargs["num_xgrid"] = num_xgrid
        elif init_mixed and model_options.get("dftbsk") is not None:
            extra_kwargs["num_xgrid"] = num_xgrid

    if init_nnenv:
        return NNENV(**model_options, **common_options)
    elif init_nnsk:
        return NNSK(**model_options["nnsk"], **common_options)
    elif init_mixed:
        return MIX(**model_options, **common_options, **extra_kwargs)
    elif init_dftbsk:
        return DFTBSK(**model_options["dftbsk"], **common_options, **extra_kwargs)
    else:
        raise ValueError("Failed to determine model type.")


def _construct_single_model_from_reference(
        checkpoint: str,
        init_nnenv: bool,
        init_nnsk: bool,
        init_mixed: bool,
        init_dftbsk: bool,
        model_options: dict,
        common_options: dict,
):
    if init_nnenv:
        return NNENV.from_reference(checkpoint, **model_options, **common_options)
    elif init_nnsk:
        return NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
    elif init_mixed:
        return MIX.from_reference(checkpoint, **model_options, **common_options)
    elif init_dftbsk:
        return DFTBSK.from_reference(checkpoint, **model_options["dftbsk"], **common_options)
    else:
        raise ValueError("Failed to determine model type.")


def _replicate_prototype_to_ensemble(
        prototype_model: nn.Module,
        distance_ranges,
        init_nnenv: bool,
        init_nnsk: bool,
        init_mixed: bool,
        init_dftbsk: bool,
        model_options: dict,
        common_options: dict,
):
    proto_state = prototype_model.state_dict()
    experts = [prototype_model]
    for _ in range(len(distance_ranges) - 1):
        m = _construct_single_model(
            init_nnenv=init_nnenv,
            init_nnsk=init_nnsk,
            init_mixed=init_mixed,
            init_dftbsk=init_dftbsk,
            model_options=model_options,
            common_options=common_options,
            ref_state_dict=proto_state,
        )
        m.load_state_dict(proto_state, strict=True)
        experts.append(m)

    return DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)


def _count_experts_in_state_dict(state_dict: dict):
    ids = set()
    if state_dict is None:
        return 0
    for k in state_dict.keys():
        if not k.startswith("experts."):
            continue
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            ids.add(int(parts[1]))
    return len(ids)


def _is_multi_expert_state_dict(state_dict: dict):
    return _count_experts_in_state_dict(state_dict) > 0


def _build_ensemble_from_wrapper_state(
        wrapper_state_dict: dict,
        distance_ranges,
        init_nnenv: bool,
        init_nnsk: bool,
        init_mixed: bool,
        init_dftbsk: bool,
        model_options: dict,
        common_options: dict,
):
    ckpt_num_experts = _count_experts_in_state_dict(wrapper_state_dict)
    if ckpt_num_experts != len(distance_ranges):
        raise ValueError(
            f"Checkpoint has {ckpt_num_experts} experts, "
            f"but current distance_ranges requires {len(distance_ranges)} experts."
        )

    experts = []
    for _ in range(len(distance_ranges)):
        m = _construct_single_model(
            init_nnenv=init_nnenv,
            init_nnsk=init_nnsk,
            init_mixed=init_mixed,
            init_dftbsk=init_dftbsk,
            model_options=model_options,
            common_options=common_options,
            ref_state_dict=wrapper_state_dict,
        )
        experts.append(m)

    model = DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)
    model.load_state_dict(wrapper_state_dict, strict=True)
    return model


# ======================================================================
# build_model
# ======================================================================
def build_model(
        checkpoint: str = None,
        model_options: dict = {},
        common_options: dict = {},
        train_options: dict = {},
        no_check: bool = False,
        device: str = None,
):
    if checkpoint is not None:
        from_scratch = False
    else:
        from_scratch = True
        if not all((model_options, common_options)):
            logging.error("You need to provide model_options and common_options when initializing from scratch.")
            raise ValueError("You need to provide model_options and common_options when initializing from scratch.")

    init_nnenv = False
    init_nnsk = False
    init_mixed = False
    init_dftbsk = False
    ckpt_state_dict = None

    if not from_scratch:
        if checkpoint.split(".")[-1] == "json":
            ckptconfig = j_loader(checkpoint)
        else:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckptconfig = f["config"]
            ckpt_state_dict = f.get("model_state_dict", None)
            del f

        if len(model_options) == 0:
            model_options = ckptconfig["model_options"]
        if len(common_options) == 0:
            common_options = ckptconfig["common_options"]
        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})
        del ckptconfig

    # 模型类型判定逻辑
    if model_options.get("dftbsk"):
        assert not model_options.get("nnsk"), "Only one of dftbsk and nnsk in model_options."
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_dftbsk = True
        else:
            raise ValueError("Model_options are not set correctly!")
    elif model_options.get("nnsk"):
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnsk = True
        else:
            raise ValueError("Model_options are not set correctly!")
    else:
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnenv = True
        else:
            raise ValueError("Model_options are not set correctly!")
        assert int(init_dftbsk) + int(init_mixed) + int(init_nnenv) + int(init_nnsk) == 1, \
            "You can only choose one mode."

    if device:
        common_options.update({"device": device})

    distance_ranges = train_options.get("distance_ranges", None)
    use_distance_ensemble = distance_ranges is not None and len(distance_ranges) > 1

    # 多专家模式
    if use_distance_ensemble:
        log.info(f"Wrapping model with DistanceEnsembleWrapper ({len(distance_ranges)} experts)")
        if from_scratch:
            prototype_model = _construct_single_model(
                init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                model_options=model_options, common_options=common_options, ref_state_dict=None,
            )
            model = _replicate_prototype_to_ensemble(
                prototype_model=prototype_model, distance_ranges=distance_ranges,
                init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                model_options=model_options, common_options=common_options,
            )
        else:
            if ckpt_state_dict is None:
                prototype_model = _construct_single_model(
                    init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                    model_options=model_options, common_options=common_options, ref_state_dict=None,
                )
                model = _replicate_prototype_to_ensemble(
                    prototype_model=prototype_model, distance_ranges=distance_ranges,
                    init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                    model_options=model_options, common_options=common_options,
                )
            elif _is_multi_expert_state_dict(ckpt_state_dict):
                model = _build_ensemble_from_wrapper_state(
                    wrapper_state_dict=ckpt_state_dict, distance_ranges=distance_ranges,
                    init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                    model_options=model_options, common_options=common_options,
                )
            else:
                prototype_model = _construct_single_model_from_reference(
                    checkpoint=checkpoint, init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed,
                    init_dftbsk=init_dftbsk, model_options=model_options, common_options=common_options,
                )
                model = _replicate_prototype_to_ensemble(
                    prototype_model=prototype_model, distance_ranges=distance_ranges,
                    init_nnenv=init_nnenv, init_nnsk=init_nnsk, init_mixed=init_mixed, init_dftbsk=init_dftbsk,
                    model_options=model_options, common_options=common_options,
                )
    # 单模型模式
    else:
        if from_scratch:
            if init_nnenv: model = NNENV(**model_options, **common_options)
            elif init_nnsk: model = NNSK(**model_options["nnsk"], **common_options)
            elif init_mixed: model = MIX(**model_options, **common_options)
            elif init_dftbsk: model = DFTBSK(**model_options["dftbsk"], **common_options)
        else:
            if init_nnenv: model = NNENV.from_reference(checkpoint, **model_options, **common_options)
            elif init_nnsk: model = NNSK.from_reference(checkpoint, **model_options["nnsk"], **common_options)
            elif init_mixed: model = MIX.from_reference(checkpoint, **model_options, **common_options)
            elif init_dftbsk: model = DFTBSK.from_reference(checkpoint, **model_options["dftbsk"], **common_options)

    if not no_check:
        for k, v in model.model_options.items():
            if k not in model_options:
                log.warning(f"The model options {k} is not defined in input model_options, set to {v}.")
            else:
                deep_dict_difference(k, v, model_options)

    model.to(model.device)
    return model

def deep_dict_difference(base_key, expected_value, model_options):
    target_dict = copy.deepcopy(model_options)
    if isinstance(expected_value, dict):
        for subk, subv in expected_value.items():
            if subk not in target_dict.get(base_key, {}):
                log.warning(
                    f"The model option {subk} in {base_key} is not defined in input model_options, set to {subv}."
                )
            else:
                target2 = copy.deepcopy(target_dict[base_key])
                deep_dict_difference(f"{subk}", subv, target2)
    else:
        if expected_value != target_dict[base_key]:
            log.warning(
                f"The model option {base_key} is set to {expected_value}, "
                f"but in input it is {target_dict[base_key]}, make sure it it correct!"
            )