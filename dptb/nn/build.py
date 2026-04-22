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
            "edge_lengths", "kpoints", "mean_max_prob", "expert_load_cv"
        }
        # 将特殊键统一纳入免缝合白名单
        for const_name in ["ATOM_TYPE_KEY", "EDGE_TYPE_KEY", "EDGE_INDEX_KEY",
                           "EDGE_VEC_KEY", "POSITIONS_KEY", "BATCH_KEY", "CELL_KEY", "PBC_KEY", "KPOINT_KEY"]:
            v = getattr(AtomicDataDict, const_name, None)
            if v is not None: keys.add(v)
        return keys

    def _get_safe_num_nodes(self, batch):
        """安全地提取节点数，严格避开 NestedTensor 带来的底层报错"""
        for key in ["ATOM_TYPE_KEY", "POSITIONS_KEY"]:
            actual_key = getattr(AtomicDataDict, key, key.lower().replace("_key", ""))
            if actual_key in batch:
                t = batch[actual_key]
                if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.ndim > 0:
                    return t.shape[0]

        # 降级策略：根据 edge_index 的最大值推断
        edge_index_key = getattr(AtomicDataDict, "EDGE_INDEX_KEY", "edge_index")
        if edge_index_key in batch:
            t = batch[edge_index_key]
            if torch.is_tensor(t) and not getattr(t, "is_nested", False) and t.numel() > 0:
                return int(t.max().item()) + 1
        return 1

    def _build_expert_masks(self, batch, expert_idx):
        dist = batch["edge_lengths"]

        d_min, d_max = self.distance_ranges[expert_idx]
        if expert_idx == self.num_experts - 1:
            edge_mask = (dist >= d_min)
        else:
            edge_mask = (dist >= d_min) & (dist < d_max)

        num_nodes = self._get_safe_num_nodes(batch)

        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=dist.device)
        if d_min > 0:
            node_mask.fill_(False)

        return edge_mask, node_mask

    def _stitch_edge_aligned_outputs(self, res, res_i, mask):
        num_edges = mask.shape[0]
        for key, src in res_i.items():
            # 1. 白名单过滤
            if key in self._excluded_stitch_keys or key not in res:
                continue

            dst = res[key]
            if not torch.is_tensor(src) or not torch.is_tensor(dst):
                continue

            # 2. 危险张量过滤：拦截 NestedTensor 和 SparseTensor
            if getattr(src, "is_nested", False) or getattr(dst, "is_nested", False):
                continue
            if getattr(src, "is_sparse", False) or getattr(dst, "is_sparse", False):
                continue

            # 3. 终极防御：如果仍有未知的奇葩张量导致 shape/掩码赋值崩溃，予以静默拦截
            try:
                if src.ndim == 0 or dst.ndim == 0 or src.shape != dst.shape:
                    continue
                if src.shape[0] != num_edges:
                    continue
                dst[mask] = src[mask]
            except Exception as e:
                log.warning(
                    f"DistanceEnsembleWrapper safely skipped stitching key '{key}' due to internal tensor error: {e}")
                continue

    def forward(self, batch):
        # 【防御 1】入口统一预处理：保证后续所有 batch copy 都携带 edge_lengths 和 vec
        if "edge_lengths" not in batch:
            batch = with_edge_vectors(batch, with_lengths=True)

        # 【防御 2】自动为单图推理补齐 PyG 依赖的所有结构键
        batch_key = getattr(AtomicDataDict, "BATCH_KEY", "batch")
        if batch_key not in batch:
            num_nodes = self._get_safe_num_nodes(batch)
            device = batch.get("edge_lengths", torch.tensor([])).device

            # 补齐 batch 数组 (全是 0，代表图 0)
            batch[batch_key] = torch.zeros(num_nodes, dtype=torch.long, device=device)
            # 补齐 ptr 游标数组 (PyG 聚合和 Scatter 常用)
            ptr_key = getattr(AtomicDataDict, "PTR_KEY", "ptr")
            batch[ptr_key] = torch.tensor([0, num_nodes], dtype=torch.long, device=device)

        expert_idx = batch.get("expert_idx", None)

        # ==================== 单专家分支 (Trainer 调用 或 单模块 Eval) ====================
        if expert_idx is not None:
            if torch.is_tensor(expert_idx):
                expert_idx_val = int(expert_idx.detach().item())
            else:
                expert_idx_val = int(expert_idx)

            clean_batch = {k: v for k, v in batch.items() if k != "expert_idx"}

            if "expert_edge_mask" not in clean_batch or "expert_node_mask" not in clean_batch:
                edge_mask, node_mask = self._build_expert_masks(clean_batch, expert_idx_val)
                clean_batch["expert_edge_mask"] = edge_mask
                clean_batch["expert_node_mask"] = node_mask

            return self.experts[expert_idx_val](clean_batch)

        # ==================== 多专家全景推理分支 (Inference Ensemble) ====================
        base_batch = batch.copy()

        # Expert 0: 获取骨架节点特征
        batch_0 = base_batch.copy()
        edge_mask_0, node_mask_0 = self._build_expert_masks(base_batch, 0)
        batch_0["expert_edge_mask"] = edge_mask_0
        batch_0["expert_node_mask"] = node_mask_0

        res = self.experts[0](batch_0)

        # Expert 1~N: 增量处理并 Stitch
        for i in range(1, self.num_experts):
            edge_mask_i, node_mask_i = self._build_expert_masks(base_batch, i)

            if not bool(edge_mask_i.any().item()):
                continue

            batch_i = base_batch.copy()
            batch_i["expert_edge_mask"] = edge_mask_i
            batch_i["expert_node_mask"] = node_mask_i

            res_i = self.experts[i](batch_i)
            self._stitch_edge_aligned_outputs(res, res_i, edge_mask_i)

        # 擦除注入的掩码，保证传出纯净的数据流
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


def _has_legacy_swiglu_s2_state(state_dict: dict) -> bool:
    if not state_dict:
        return False
    return any(".activation.mul." in key for key in state_dict.keys())


def _maybe_enable_legacy_swiglu_s2_compat(model_options: dict, state_dict: dict):
    if not state_dict or not model_options:
        return model_options

    embedding = model_options.get("embedding", None)
    if not isinstance(embedding, dict):
        return model_options
    if embedding.get("method") not in {"lem_moe_v3", "lem_moe_v3_h0"}:
        return model_options
    if embedding.get("swiglu_s2_compat_mode", "modern") != "modern":
        return model_options
    if not _has_legacy_swiglu_s2_state(state_dict):
        return model_options

    patched = copy.deepcopy(model_options)
    patched.setdefault("embedding", {})
    patched["embedding"]["swiglu_s2_compat_mode"] = "legacy_uniform_only"
    log.warning(
        "Detected legacy flat SwiGLU-S2 checkpoint layout; forcing "
        "embedding.swiglu_s2_compat_mode='legacy_uniform_only' for compatibility."
    )
    return patched


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
        train_options: dict = {},
        no_check: bool = False,
        device: str = None,
):
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
    ckpt_state_dict = None

    if not from_scratch:
        if checkpoint.split(".")[-1] == "json":
            ckptconfig = j_loader(checkpoint)
        else:
            f = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckptconfig = f['config']
            ckpt_state_dict = f.get("model_state_dict", None)
            del f

        if len(model_options) == 0:
            model_options = ckptconfig["model_options"]

        if len(common_options) == 0:
            common_options = ckptconfig["common_options"]

        if len(train_options) == 0:
            train_options = ckptconfig.get("train_options", {})

        del ckptconfig

    model_options = _maybe_enable_legacy_swiglu_s2_compat(model_options, ckpt_state_dict)

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
            raise ValueError("Model_options are not set correctly!")
    else:
        if all((model_options.get("embedding"), model_options.get("prediction"))):
            init_nnenv = True
            if model_options["prediction"]['method'] == 'sktb':
                if not model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method must be se2 for sktb prediction in nnenv mode.")
                    raise ValueError("The embedding method must be se2 for sktb prediction in deeptb mode.")

            if model_options["prediction"]['method'] == 'e3tb':
                if model_options['embedding']['method'] in ['se2']:
                    log.error("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
                    raise ValueError("The embedding method can not be se2 for e3tb prediction in deeptb mode.")
        else:
            raise ValueError("Model_options are not set correctly!")

        assert int(init_dftbsk) + int(init_mixed) + int(init_nnenv) + int(
            init_nnsk) == 1, "IF not sk, you can only choose one of the mixed, nnenv, dftb and nnsk options."

    if device:
        common_options.update({"device": device})

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
