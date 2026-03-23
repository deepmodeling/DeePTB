from dptb.nn.deeptb import NNENV, MIX
import logging
from dptb.nn.nnsk import NNSK
from dptb.nn.dftbsk import DFTBSK
import torch
import torch.nn as nn
from dptb.utils.tools import j_must_have, j_loader
from dptb.data import AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors
import copy
import random
import numpy as np

log = logging.getLogger(__name__)


# ======================================================================
# [独立扩展模块] Deterministic Seed Context Manager
# ======================================================================
class DeterministicExpertSeed:
    """
    上下文管理器：精准为当前的 Expert 设定固定的初始化种子（保证 Windows/Linux 完美对齐）。
    退出时自动恢复之前的全局随机状态，避免污染 DataLoader 等后续流程。
    """

    def __init__(self, seed_val: int):
        self.seed_val = seed_val

    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.get_rng_state_all()

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_val)
            torch.cuda.manual_seed_all(self.seed_val)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_state)


# ======================================================================
# [独立扩展模块] Distance Ensemble Wrapper
# ======================================================================
class DistanceEnsembleWrapper(nn.Module):
    def __init__(self, experts, distance_ranges):
        super().__init__()
        assert len(experts) == len(distance_ranges), \
            f"len(experts) != len(distance_ranges): {len(experts)} vs {len(distance_ranges)}"

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.experts = nn.ModuleList(experts)

        base_model = self.experts[0]
        self.name = getattr(base_model, "name", "distance_ensemble")
        self.device = getattr(base_model, "device", torch.device("cpu"))
        self.dtype = getattr(base_model, "dtype", torch.float32)
        self.model_options = copy.deepcopy(getattr(base_model, "model_options", {}))
        self._excluded_stitch_keys = self._build_excluded_stitch_keys()

    def __getattr__(self, name):
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
            "expert_idx", "expert_edge_mask", "expert_node_mask",
            "__slices__", "__cumsum__", "__cat_dims__", "__num_nodes_list__",
            "__data_class__", "edge_vec", "edge_index", "edge_type",
            "atom_types", "batch", "ptr", "cell", "pbc", "pos", "positions",
            "edge_lengths"
        }
        for const_name in ["ATOM_TYPE_KEY", "EDGE_TYPE_KEY", "EDGE_INDEX_KEY",
                           "EDGE_VEC_KEY", "POSITIONS_KEY", "BATCH_KEY", "CELL_KEY", "PBC_KEY"]:
            v = getattr(AtomicDataDict, const_name, None)
            if v is not None: keys.add(v)
        return keys

    def _build_expert_masks(self, batch, expert_idx):
        # 此时 batch 必然已包含 edge_lengths (由 forward 入口保证)
        dist = batch["edge_lengths"]

        d_min, d_max = self.distance_ranges[expert_idx]
        if expert_idx == self.num_experts - 1:
            edge_mask = (dist >= d_min)
        else:
            edge_mask = (dist >= d_min) & (dist < d_max)

        # 安全获取节点总数以生成 node_mask
        if getattr(AtomicDataDict, "ATOM_TYPE_KEY", "atom_types") in batch:
            num_nodes = batch[getattr(AtomicDataDict, "ATOM_TYPE_KEY", "atom_types")].shape[0]
        elif getattr(AtomicDataDict, "POSITIONS_KEY", "pos") in batch:
            num_nodes = batch[getattr(AtomicDataDict, "POSITIONS_KEY", "pos")].shape[0]
        else:
            num_nodes = dist.shape[0]  # Fallback

        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=dist.device)
        if d_min > 0:
            node_mask.fill_(False)

        return edge_mask, node_mask

    def _stitch_edge_aligned_outputs(self, res, res_i, mask):
        num_edges = mask.shape[0]
        for key, src in res_i.items():
            if key in self._excluded_stitch_keys or key not in res: continue
            dst = res[key]
            if not torch.is_tensor(src) or not torch.is_tensor(dst): continue
            if src.ndim == 0 or dst.ndim == 0 or src.shape != dst.shape: continue
            if src.shape[0] != num_edges: continue
            dst[mask] = src[mask]

    def forward(self, batch):
        # 【防御 1】入口统一预处理：保证后续所有 batch copy 都携带 edge_lengths 和 vec
        # 避免 expert 内部再次执行 with_edge_vectors 导致开销或缺输入崩溃
        if "edge_lengths" not in batch:
            batch = with_edge_vectors(batch, with_lengths=True)

        expert_idx = batch.get("expert_idx", None)

        # ==================== 单专家分支 (Trainer 调用 或 单模块 Eval) ====================
        if expert_idx is not None:
            if torch.is_tensor(expert_idx):
                expert_idx_val = int(expert_idx.detach().item())
            else:
                expert_idx_val = int(expert_idx)

            # 剥离引发 TorchScript 严格类型检查的 int
            clean_batch = {k: v for k, v in batch.items() if k != "expert_idx"}

            # 【防御 2】自动补齐 Mask：如果在 Inference 阶段手动指定单专家，外部可能没给 mask
            if "expert_edge_mask" not in clean_batch or "expert_node_mask" not in clean_batch:
                edge_mask, node_mask = self._build_expert_masks(clean_batch, expert_idx_val)
                clean_batch["expert_edge_mask"] = edge_mask
                clean_batch["expert_node_mask"] = node_mask

            return self.experts[expert_idx_val](clean_batch)

        # ==================== 多专家全景推理分支 (Inference Ensemble) ====================

        # 使用基础拷贝隔离外部数据
        base_batch = batch.copy()

        # Expert 0：作为骨架必须完整执行，获取全量 node_features
        batch_0 = base_batch.copy()
        edge_mask_0, node_mask_0 = self._build_expert_masks(base_batch, 0)
        batch_0["expert_edge_mask"] = edge_mask_0
        batch_0["expert_node_mask"] = node_mask_0

        res = self.experts[0](batch_0)

        # Expert 1~N：增量执行并 Stitch
        for i in range(1, self.num_experts):
            edge_mask_i, node_mask_i = self._build_expert_masks(base_batch, i)

            # 【防御 3】显式 .item() 判断，防止 Tensor Bool Ambiguity 以及隐式的 Graph Break
            if not bool(edge_mask_i.any().item()):
                continue

            # 字典浅拷贝阻断 key 级别的值覆盖（防止后面的 expert 强行把 node_features 改成 None）
            batch_i = base_batch.copy()
            batch_i["expert_edge_mask"] = edge_mask_i
            batch_i["expert_node_mask"] = node_mask_i

            res_i = self.experts[i](batch_i)
            self._stitch_edge_aligned_outputs(res, res_i, edge_mask_i)

        # 【防御 4】剥离注入的 Mask，确保返回的输出如同单一模型一样纯净
        # 避免下游 Loss 函数看到 Mask 后只计算局部 Loss
        if "expert_edge_mask" in res:
            del res["expert_edge_mask"]
        if "expert_node_mask" in res:
            del res["expert_node_mask"]

        return res
# ======================================================================
# [独立扩展模块] Multi-Expert Helper Functions
# ======================================================================
def _infer_num_xgrid_from_state_dict(state_dict: dict):
    if state_dict is None: return None
    for k, v in state_dict.items():
        if k.endswith("distance_param") and torch.is_tensor(v) and v.ndim >= 1: return int(v.shape[0])
    return None


def _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                            ref_state_dict=None):
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


def _construct_single_model_from_reference(checkpoint, init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options,
                                           common_options):
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


def _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk, init_mixed, init_dftbsk,
                                     model_options, common_options):
    proto_state = prototype_model.state_dict()
    experts = [prototype_model]
    for i in range(1, len(distance_ranges)):
        with DeterministicExpertSeed(i + 1):
            m = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                                        ref_state_dict=proto_state)
        m.load_state_dict(proto_state, strict=True)
        experts.append(m)
    return DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)


def _count_experts_in_state_dict(state_dict: dict):
    if state_dict is None: return 0
    ids = {int(k.split(".")[1]) for k in state_dict.keys() if
           k.startswith("experts.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()}
    return len(ids)


def _is_multi_expert_state_dict(state_dict: dict):
    return _count_experts_in_state_dict(state_dict) > 0


def _build_ensemble_from_wrapper_state(wrapper_state_dict, distance_ranges, init_nnenv, init_nnsk, init_mixed,
                                       init_dftbsk, model_options, common_options):
    ckpt_num_experts = _count_experts_in_state_dict(wrapper_state_dict)
    if ckpt_num_experts != len(distance_ranges):
        raise ValueError(f"Checkpoint has {ckpt_num_experts} experts, but requires {len(distance_ranges)}.")
    experts = []
    for i in range(len(distance_ranges)):
        with DeterministicExpertSeed(i + 1):
            m = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options, common_options,
                                        ref_state_dict=wrapper_state_dict)
        experts.append(m)
    model = DistanceEnsembleWrapper(experts=experts, distance_ranges=distance_ranges)
    model.load_state_dict(wrapper_state_dict, strict=True)
    return model


# ======================================================================
# [核心主程序] 原版 build_model
# ======================================================================
def build_model(
        checkpoint: str = None,
        model_options: dict = {},
        common_options: dict = {},
        train_options: dict = {},  # 新增参数，支持外部传入 distance_ranges
        no_check: bool = False,
        device: str = None,
):
    """
    The build model method should composed of the following steps:
        1. process the configs from user input and the config from the checkpoint (if any).
        2. construct the model based on the configs.
        3. process the config dict for the output dict.
        run_opt = {
        "init_model": init_model,
        "restart": restart,
    }
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
    ckpt_state_dict = None  # 新增变量用于拦截 multi-expert 权重

    if not from_scratch:
        if checkpoint.split(".")[-1] == "json":
            ckptconfig = j_loader(checkpoint)
        else:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckptconfig = f['config']
            ckpt_state_dict = f.get("model_state_dict", None)  # 拦截权重
            del f

        if len(model_options) == 0:
            model_options = ckptconfig["model_options"]

        if len(common_options) == 0:
            common_options = ckptconfig["common_options"]

        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})  # 拦截 train_options

        del ckptconfig

    # ================= 原版严格的类型判断逻辑 =================
    if model_options.get("dftbsk"):
        assert not model_options.get("nnsk"), "There should only be one of the dftbsk and nnsk in model_options."
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                log.error("The prediction method must be sktb for mix mode.")
                raise ValueError("The prediction method must be sktb for mix mode.")

            if not model_options['embedding']['method'] in ['se2']:
                log.error("The embedding method must be se2 for mix mode.")
                raise ValueError("The embedding method must be se2 for mix mode.")
        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_dftbsk = True
        else:
            log.error("Model_options are not set correctly! \n" +
                      "You can only choose one of the nnsk, dftb, mixed and nnenv modes.\n" +
                      " -  `mixed`, set all the `nnsk` or `dftbsk` and both `embedding` and `prediction` options.\n" +
                      " -  `nnenv`, set `embedding` and `prediction` options and no `nnsk` and no `dftbsk`.\n" +
                      " -  `nnsk`, set only `nnsk` options.\n" +
                      " -  `dftbsk`, set only `dftbsk` options.")
            raise ValueError("Model_options are not set correctly!")

    elif model_options.get("nnsk"):
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_mixed = True
            if not model_options['prediction']['method'] == 'sktb':
                log.error("The prediction method must be sktb for mix mode.")
                raise ValueError("The prediction method must be sktb for mix mode.")

            if not model_options['embedding']['method'] in ['se2']:
                log.error("The embedding method must be se2 for mix mode.")
                raise ValueError("The embedding method must be se2 for mix mode.")

        elif not any((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnsk = True
        else:
            log.error("Model_options are not set correctly! \n" +
                      "You can only choose one of the nnsk, dftb, mixed and nnenv modes.\n" +
                      " -  `mixed`, set all the `nnsk` or `dftbsk` and both `embedding` and `prediction` options.\n" +
                      " -  `nnenv`, set `embedding` and `prediction` options and no `nnsk` and no `dftbsk`.\n" +
                      " -  `nnsk`, set only `nnsk` options.\n" +
                      " -  `dftbsk`, set only `dftbsk` options.")
            raise ValueError("Model_options are not set correctly!")
    else:
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnenv = True
            if model_options["prediction"]['method'] == 'sktb':
                log.warning(
                    "The prediction method is sktb, but the nnsk option is not set. this is highly not recommand.\n" +
                    "We recommand to train nnsk then train mix model for sktb. \n" +
                    "Or you can use the dftb + nnenv to train a mix model for sktb. \n" +
                    "Please make sure you know what you are doing!")
                if not model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method must be se2 for sktb prediction in nnenv mode.")
                    raise ValueError("The embedding method must be se2 for sktb prediction in deeptb mode.")

            if model_options["prediction"]['method'] == 'e3tb':
                if model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
                    raise ValueError("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
        else:
            log.error("Model_options are not set correctly! \n" +
                      "You can only choose one of the nnsk, dftb, mixed and nnenv modes.\n" +
                      " -  `mixed`, set all the `nnsk` or `dftbsk` and both `embedding` and `prediction` options.\n" +
                      " -  `nnenv`, set `embedding` and `prediction` options and no `nnsk` and no `dftbsk`.\n" +
                      " -  `nnsk`, set only `nnsk` options.\n" +
                      " -  `dftbsk`, set only `dftbsk` options.")
            raise ValueError("Model_options are not set correctly!")

        assert int(init_dftbsk) + int(init_mixed) + int(init_nnenv) + int(
            init_nnsk) == 1, "IF not sk, you can only choose one of the mixed, nnenv, dftb and nnsk options."

    if device:
        common_options.update({"device": device})

    # ================= 原版/多专家 逻辑分叉点 =================
    distance_ranges = train_options.get("distance_ranges", None)
    use_distance_ensemble = distance_ranges is not None and len(distance_ranges) > 1

    if use_distance_ensemble:
        log.info(f"Wrapping model with DistanceEnsembleWrapper ({len(distance_ranges)} experts)")
        if from_scratch:
            with DeterministicExpertSeed(1):
                prototype_model = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk, model_options,
                                                          common_options, ref_state_dict=None)
            model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                     init_mixed, init_dftbsk, model_options, common_options)
        else:
            if ckpt_state_dict is None:
                with DeterministicExpertSeed(1):
                    prototype_model = _construct_single_model(init_nnenv, init_nnsk, init_mixed, init_dftbsk,
                                                              model_options, common_options, ref_state_dict=None)
                model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                         init_mixed, init_dftbsk, model_options, common_options)
            elif _is_multi_expert_state_dict(ckpt_state_dict):
                model = _build_ensemble_from_wrapper_state(ckpt_state_dict, distance_ranges, init_nnenv, init_nnsk,
                                                           init_mixed, init_dftbsk, model_options, common_options)
            else:
                with DeterministicExpertSeed(1):
                    prototype_model = _construct_single_model_from_reference(checkpoint, init_nnenv, init_nnsk,
                                                                             init_mixed, init_dftbsk, model_options,
                                                                             common_options)
                model = _replicate_prototype_to_ensemble(prototype_model, distance_ranges, init_nnenv, init_nnsk,
                                                         init_mixed, init_dftbsk, model_options, common_options)

    else:
        # 完全保留原版的逻辑，仅包裹 Seed 以防止 Windows/Linux 差异导致单模型无法复现
        with DeterministicExpertSeed(1):
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
                print(common_options)
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
    # =======================================================

    if not no_check:
        for k, v in model.model_options.items():
            if k not in model_options:
                log.warning(f"The model options {k} is not defined in input model_options, set to {v}.")
            else:
                deep_dict_difference(k, v, model_options)

    model.to(model.device)

    return model


def deep_dict_difference(base_key, expected_value, model_options):
    """
    递归地记录嵌套字典中的选项差异。

    :param base_key: 基础键名，用于构建警告消息的前缀。
    :param expected_value: 期望的值，可能是字典或非字典类型。
    :param model_options: 用于比较的模型选项字典。
    """
    target_dict = copy.deepcopy(model_options)  # 防止修改原始字典
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