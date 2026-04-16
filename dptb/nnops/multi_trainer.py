import contextlib
import time
import logging
import copy
import heapq
import os
from typing import Union, Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.profiler import profile as torch_profile, ProfilerActivity

from dptb.utils.tools import get_lr_scheduler, get_optimizer
from dptb.data import AtomicDataset, AtomicData, DataLoader
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model

log = logging.getLogger(__name__)


class _StageTagger:
    def __init__(self, trainer, enabled: bool, freq: int, cuda_mem: bool, cuda_sync: bool, oom_dump: bool):
        self.trainer = trainer
        self.enabled = bool(enabled)
        self.freq = max(int(freq), 1)
        self.cuda_mem = bool(cuda_mem)
        self.cuda_sync = bool(cuda_sync)
        self.oom_dump = bool(oom_dump)

    def _device(self) -> torch.device:
        return self.trainer.device if isinstance(self.trainer.device, torch.device) else torch.device(self.trainer.device)

    def _is_cuda(self) -> bool:
        dev = self._device()
        return torch.cuda.is_available() and dev.type == "cuda"

    def _cuda_mem(self):
        if not self._is_cuda():
            return None
        dev = self._device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        free, total = torch.cuda.mem_get_info(dev)
        return alloc, reserved, peak, free, total

    def _fmt_mem(self, mem):
        if mem is None:
            return ""
        alloc, reserved, peak, free, total = mem
        mb = 1024 ** 2
        return (
            f" | cuda_alloc={alloc/mb:.1f}MB"
            f" cuda_reserved={reserved/mb:.1f}MB"
            f" cuda_peak={peak/mb:.1f}MB"
            f" free={free/mb:.1f}MB total={total/mb:.1f}MB"
        )

    def dump_cuda_mem_summary(self, where: str):
        if not self._is_cuda():
            return
        dev = self._device()
        mb = 1024 ** 2
        alloc = torch.cuda.memory_allocated(dev) / mb
        reserved = torch.cuda.memory_reserved(dev) / mb
        peak = torch.cuda.max_memory_allocated(dev) / mb
        free, total = torch.cuda.mem_get_info(dev)
        log.error(
            f"[OOM-DUMP] where={where} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB "
            f"free={free/mb:.1f}MB total={total/mb:.1f}MB"
        )
        if self.oom_dump:
            try:
                log.error("[OOM-DUMP] memory_summary:\n%s", torch.cuda.memory_summary(dev, abbreviated=False))
            except Exception:
                pass

    @contextlib.contextmanager
    def tag(self, name: str, *, it: Optional[int] = None, expert: Optional[int] = None, extra: str = ""):
        if not self.enabled:
            yield
            return
        if it is not None and (it % self.freq != 0):
            yield
            return

        prefix = f"[TAG][it={it}]"
        if expert is not None:
            prefix += f"[expert={expert}]"
        prefix += f"[{name}]"

        nvtx_pushed = False
        if self._is_cuda():
            try:
                torch.cuda.nvtx.range_push(f"{prefix}{(' ' + extra) if extra else ''}")
                nvtx_pushed = True
            except Exception:
                nvtx_pushed = False

        dev = self._device()
        if self.cuda_mem and self._is_cuda():
            try:
                torch.cuda.reset_peak_memory_stats(dev)
            except Exception:
                pass

        t0 = time.perf_counter()

        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.dump_cuda_mem_summary(where=f"{name} it={it} expert={expert}")
            raise
        finally:
            if self.cuda_sync and self._is_cuda():
                torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            mem1 = self._cuda_mem() if self.cuda_mem else None
            log.info(f"{prefix} dt={dt:.4f}s{self._fmt_mem(mem1)}{(' | ' + extra) if extra else ''}")

            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


class MultiTrainer(Trainer):
    object_keys = ["lr_schedulers", "optimizers"]

    _P_LOSS_OPT_SUM = 0
    _P_ONSITE_WEIGHTED_SUM = 1
    _P_HOPPING_WEIGHTED_SUM = 2
    _P_ACTIVE_NODES_SUM = 3
    _P_ACTIVE_EDGES_SUM = 4
    _P_ONSITE_L1_SUM = 5
    _P_ONSITE_MSE_SUM = 6
    _P_ONSITE_CNT_SUM = 7
    _P_HOPPING_L1_SUM = 8
    _P_HOPPING_MSE_SUM = 9
    _P_HOPPING_CNT_SUM = 10
    _P_Z_SUM = 11
    _P_Z_CNT = 12
    _P_CV_SUM = 13
    _P_CV_CNT = 14
    _P_GRAD_NORM_SUM = 15
    _P_STEP_COUNT = 16
    _PACK_LEN = 17

    def __init__(
        self,
        distance_ranges: list,
        train_options: dict,
        common_options: dict,
        model: torch.nn.Module,
        train_datasets: AtomicDataset,
        reference_datasets: Union[AtomicDataset, None] = None,
        validation_datasets: Union[AtomicDataset, None] = None,
        distributed_expert: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        trainer_common_options = copy.deepcopy(common_options)
        if distributed_expert:
            trainer_common_options["device"] = "cpu"

        super().__init__(
            train_options=train_options,
            common_options=trainer_common_options,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )
        self.common_options = common_options
        if self.use_reference:
            self.reference_datasets = getattr(self, "reference_datesets", None)
        else:
            self.reference_datasets = None

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)

        self.distributed_expert = bool(distributed_expert)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_expert_idx = self.rank if self.distributed_expert else None
        self.is_main_process = (not self.distributed_expert) or (self.rank == 0)
        if self.distributed_expert:
            self.device = common_options["device"]
            self._move_aux_modules_to_device(self._device_obj())

        self.parallel_multi = bool(self.train_options.get("parallel_multi", False))
        if self.distributed_expert:
            if self.world_size != self.num_experts:
                raise ValueError(
                    f"distributed_expert mode requires world_size == num_experts, "
                    f"got world_size={self.world_size}, num_experts={self.num_experts}"
                )
            if self.parallel_multi:
                log.warning("distributed_expert=True: parallel_multi will be disabled.")
            self.parallel_multi = False
            self.train_options["parallel_multi"] = False

        self.debug_tags = bool(self.train_options.get("debug_tags", False))
        self.debug_tag_freq = int(self.train_options.get("debug_tag_freq", 1))
        self.debug_tag_cuda_mem = bool(self.train_options.get("debug_tag_cuda_mem", True))
        self.debug_tag_cuda_sync = bool(self.train_options.get("debug_tag_cuda_sync", False))
        self.debug_oom_dump = bool(self.train_options.get("debug_oom_dump", True))

        self.debug_profile = bool(self.train_options.get("debug_profile", False))
        self.debug_profile_start_iter = int(self.train_options.get("debug_profile_start_iter", 5))
        self.debug_profile_end_iter = int(
            self.train_options.get("debug_profile_end_iter", self.debug_profile_start_iter)
        )
        self.debug_profile_dir = self.train_options.get("debug_profile_dir", None)

        self.display_sync_freq = max(int(self.train_options.get("display_freq", 1)), 1)
        self.distributed_rank0_prepare_batch = bool(
            self.train_options.get("distributed_rank0_prepare_batch", False)
        )

        # dataloader options
        self.train_num_workers = int(self.train_options.get("train_num_workers", self.train_options.get("num_workers", 0)))
        self.ref_num_workers = int(self.train_options.get("ref_num_workers", self.train_num_workers))
        self.val_num_workers = int(self.train_options.get("val_num_workers", self.train_num_workers))

        dev_obj = self._device_obj()
        self.data_pin_memory = bool(self.train_options.get("data_pin_memory", dev_obj.type == "cuda"))

        self.data_persistent_workers = bool(self.train_options.get("data_persistent_workers", self.train_num_workers > 0))
        self.data_prefetch_factor = int(self.train_options.get("data_prefetch_factor", 2))

        self._tagger = _StageTagger(
            trainer=self,
            enabled=self.debug_tags,
            freq=self.debug_tag_freq,
            cuda_mem=self.debug_tag_cuda_mem,
            cuda_sync=self.debug_tag_cuda_sync,
            oom_dump=self.debug_oom_dump,
        )

        self.log_single_model_compatible_loss = bool(
            self.train_options.get("log_single_model_compatible_loss", True)
        )
        self.log_single_model_compatible_loss_mode = str(
            self.train_options.get("log_single_model_compatible_loss_mode", "reduce")
        ).lower()

        if self.distributed_expert and self.log_single_model_compatible_loss_mode == "full_forward":
            log.warning(
                "distributed_expert=True does not support full stitched forward across GPUs. "
                "Fallback log_single_model_compatible_loss_mode from 'full_forward' to 'reduce'."
            )
            self.log_single_model_compatible_loss_mode = "reduce"

        # ---------------- per-expert optimizer / scheduler overrides ----------------
        self.expert_lrs = self._parse_expert_lrs(self.train_options.get("expert_lrs", None))
        self.expert_optimizer_overrides = self._parse_expert_config_overrides(
            self.train_options.get("expert_optimizer_overrides", None),
            field_name="train_options.expert_optimizer_overrides",
        )
        self.expert_lr_scheduler_overrides = self._parse_expert_config_overrides(
            self.train_options.get("expert_lr_scheduler_overrides", None),
            field_name="train_options.expert_lr_scheduler_overrides",
        )
        # ----------------------------------------------------------------------------

        log.info(
            f"[MultiTrainer][rank={self.rank}] num_experts={self.num_experts}, "
            f"distributed_expert={self.distributed_expert}, parallel_multi={self.parallel_multi}, "
            f"display_sync_freq={self.display_sync_freq}, "
            f"distributed_rank0_prepare_batch={self.distributed_rank0_prepare_batch}, "
            f"train_num_workers={self.train_num_workers}, ref_num_workers={self.ref_num_workers}, val_num_workers={self.val_num_workers}, "
            f"pin_memory={self.data_pin_memory}, persistent_workers={self.data_persistent_workers}, prefetch_factor={self.data_prefetch_factor}, "
            f"log_single_model_compatible_loss={self.log_single_model_compatible_loss}, "
            f"mode={self.log_single_model_compatible_loss_mode}, "
            f"expert_lrs={'(default optimizer.lr)' if self.expert_lrs is None else self.expert_lrs}, "
            f"expert_optimizer_overrides={self._summarize_expert_override_list(self.expert_optimizer_overrides)}, "
            f"expert_lr_scheduler_overrides={self._summarize_expert_override_list(self.expert_lr_scheduler_overrides)}."
        )

        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

        if self.distributed_expert:
            self._materialize_local_expert_only()

        def _make_opt_for_expert(i: int):
            opt_cfg = self._build_optimizer_cfg_for_expert(i)
            return get_optimizer(model_param=self.model.experts[i].parameters(), **opt_cfg)

        def _make_scheduler_for_expert(i: int, opt):
            sch_cfg = self._build_lr_scheduler_cfg_for_expert(i)
            return get_lr_scheduler(optimizer=opt, **sch_cfg)

        if self.distributed_expert:
            self.optimizers = [None] * self.num_experts
            self.lr_schedulers = [None] * self.num_experts
            idx = self.local_expert_idx

            opt = _make_opt_for_expert(idx)
            sch = _make_scheduler_for_expert(idx, opt)
            self.optimizers[idx] = opt
            self.lr_schedulers[idx] = sch
        else:
            self.optimizers = []
            self.lr_schedulers = []
            for i in range(self.num_experts):
                opt = _make_opt_for_expert(i)
                sch = _make_scheduler_for_expert(i, opt)
                self.optimizers.append(opt)
                self.lr_schedulers.append(sch)

        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "lr_scheduler"):
            del self.lr_scheduler

        self._maybe_rebuild_loaders_in_multi_trainer()
        self._warn_non_expert_trainables()
        self._t_last_iter_end: Optional[float] = None
        self._reset_display_window_buffers()

    # ---------------- per-expert optimizer / scheduler parsing & checks ----------------
    def _parse_expert_lrs(self, expert_lrs) -> Optional[List[float]]:
        """
        train_options.expert_lrs:
          - None / []: disabled
          - list/tuple of float with length == num_experts: enabled
        """
        if expert_lrs is None:
            return None
        if isinstance(expert_lrs, (list, tuple)):
            if len(expert_lrs) == 0:
                return None
            if len(expert_lrs) != self.num_experts:
                raise ValueError(
                    f"train_options.expert_lrs length must match num_experts={self.num_experts}, "
                    f"got len(expert_lrs)={len(expert_lrs)}"
                )
            lrs = [float(x) for x in expert_lrs]
            bad = [i for i, lr in enumerate(lrs) if not (lr > 0.0)]
            if bad:
                raise ValueError(f"train_options.expert_lrs must be all > 0.0, bad indices: {bad}, values={lrs}")
            return lrs
        raise TypeError(f"train_options.expert_lrs must be a list/tuple of float (or empty/None), got {type(expert_lrs)}")

    def _parse_expert_config_overrides(self, overrides, field_name: str) -> Optional[List[Dict[str, Any]]]:
        if overrides is None:
            return None
        if isinstance(overrides, (list, tuple)):
            if len(overrides) == 0:
                return None
            if len(overrides) != self.num_experts:
                raise ValueError(
                    f"{field_name} length must match num_experts={self.num_experts}, "
                    f"got len({field_name})={len(overrides)}"
                )
            parsed = []
            for idx, item in enumerate(overrides):
                if item is None:
                    parsed.append({})
                elif isinstance(item, dict):
                    parsed.append(copy.deepcopy(item))
                else:
                    raise TypeError(
                        f"{field_name}[{idx}] must be dict or null/None, got {type(item)}"
                    )
            return parsed
        raise TypeError(f"{field_name} must be a list/tuple of dict (or empty/None), got {type(overrides)}")

    @staticmethod
    def _summarize_expert_override_list(overrides: Optional[List[Dict[str, Any]]]) -> str:
        if overrides is None:
            return "(shared base config)"
        active = [idx for idx, item in enumerate(overrides) if item]
        if not active:
            return "(shared base config)"
        return f"active_experts={active}"

    def _build_optimizer_cfg_for_expert(self, expert_idx: int) -> Dict[str, Any]:
        opt_cfg = copy.deepcopy(self.train_options["optimizer"])
        opt_override = None
        if self.expert_optimizer_overrides is not None:
            opt_override = self.expert_optimizer_overrides[expert_idx]
            opt_cfg.update(opt_override)
        if self.expert_lrs is not None:
            lr_overridden_in_opt = isinstance(opt_override, dict) and ("lr" in opt_override)
            if not lr_overridden_in_opt:
                opt_cfg["lr"] = float(self.expert_lrs[expert_idx])
        return opt_cfg

    def _build_lr_scheduler_cfg_for_expert(self, expert_idx: int) -> Dict[str, Any]:
        sch_cfg = copy.deepcopy(self.train_options["lr_scheduler"])
        if self.expert_lr_scheduler_overrides is not None:
            sch_cfg.update(self.expert_lr_scheduler_overrides[expert_idx])
        return sch_cfg
    # -----------------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # dataloader rebuild in MultiTrainer only
    # ---------------------------------------------------------------------

    def _make_loader_compat(self, dataset, batch_size, shuffle, num_workers):
        kwargs = {"num_workers": int(num_workers)}
        if kwargs["num_workers"] > 0:
            kwargs["pin_memory"] = self.data_pin_memory
            kwargs["persistent_workers"] = self.data_persistent_workers
            kwargs["prefetch_factor"] = self.data_prefetch_factor
        else:
            kwargs["pin_memory"] = self.data_pin_memory

        trial_kwargs = [
            kwargs,
            {k: v for k, v in kwargs.items() if k != "prefetch_factor"},
            {k: v for k, v in kwargs.items() if k not in ("prefetch_factor", "persistent_workers")},
            {k: v for k, v in kwargs.items() if k != "pin_memory"},
            {},
        ]

        last_err = None
        for kw in trial_kwargs:
            try:
                return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kw)
            except TypeError as e:
                last_err = e
                continue
        raise last_err

    def _maybe_rebuild_loaders_in_multi_trainer(self):
        worker_keys = {
            "train_num_workers", "ref_num_workers", "val_num_workers",
            "data_pin_memory", "data_persistent_workers", "data_prefetch_factor"
        }
        need_rebuild = (
            self.distributed_expert or
            self.distributed_rank0_prepare_batch or
            any(k in self.train_options for k in worker_keys)
        )

        if not need_rebuild:
            return

        train_workers = self.train_num_workers
        ref_workers = self.ref_num_workers
        val_workers = self.val_num_workers

        if self.distributed_expert and self.distributed_rank0_prepare_batch and self.rank != 0:
            train_workers = 0
            ref_workers = 0
            val_workers = 0

        self.train_loader = self._make_loader_compat(
            dataset=self.train_datasets,
            batch_size=self.train_options["batch_size"],
            shuffle=True,
            num_workers=train_workers,
        )

        if self.use_reference:
            self.reference_loader = self._make_loader_compat(
                dataset=self.reference_datasets,
                batch_size=self.train_options["ref_batch_size"],
                shuffle=True,
                num_workers=ref_workers,
            )

        if self.use_validation:
            self.validation_loader = self._make_loader_compat(
                dataset=self.validation_datasets,
                batch_size=self.train_options["val_batch_size"],
                shuffle=not self.distributed_expert,
                num_workers=val_workers,
            )

        log.info(
            f"[MultiTrainer][rank={self.rank}] rebuilt loaders in MultiTrainer: "
            f"train_workers={train_workers}, ref_workers={ref_workers}, val_workers={val_workers}"
        )

    # ---------------------------------------------------------------------
    # dist helpers
    # ---------------------------------------------------------------------

    def _dist_ready(self):
        return self.distributed_expert and dist.is_available() and dist.is_initialized()

    def _device_obj(self):
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda_device(self):
        return self._device_obj().type == "cuda" and torch.cuda.is_available()

    def _use_cuda_stream_parallel(self):
        return (not self.distributed_expert) and self.parallel_multi and self.num_experts > 1 and self._is_cuda_device()

    def _all_reduce_(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM, name: str = "dist/all_reduce"):
        if self._dist_ready():
            with self._tagger.tag(name, it=self.iter, extra=f"numel={tensor.numel()}"):
                dist.all_reduce(tensor, op=op)
        return tensor

    def _all_gather_(self, output_list: List[torch.Tensor], tensor: torch.Tensor, name: str = "dist/all_gather"):
        if self._dist_ready():
            with self._tagger.tag(name, it=self.iter, extra=f"numel={tensor.numel()} world={self.world_size}"):
                dist.all_gather(output_list, tensor)
        else:
            output_list[0].copy_(tensor)
        return output_list

    def _recursive_set_device_attr(self, module: nn.Module, device: torch.device):
        for m in module.modules():
            if hasattr(m, "device"):
                try:
                    setattr(m, "device", device)
                except Exception:
                    pass

    def _move_aux_modules_to_device(self, device: torch.device):
        for attr in ("train_lossfunc", "validation_lossfunc", "reference_lossfunc"):
            module = getattr(self, attr, None)
            if module is None:
                continue
            if isinstance(module, nn.Module):
                module.to(device)
                self._recursive_set_device_attr(module, device)
            elif hasattr(module, "device"):
                try:
                    setattr(module, "device", device)
                except Exception:
                    pass

    def _materialize_local_expert_only(self):
        local_dev = self._device_obj()
        cpu_dev = torch.device("cpu")

        for i, expert in enumerate(self.model.experts):
            target = local_dev if i == self.local_expert_idx else cpu_dev
            expert.to(target)
            self._recursive_set_device_attr(expert, target)

        if hasattr(self.model, "device"):
            self.model.device = local_dev

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(
            f"[MultiTrainer][rank={self.rank}] local_expert_idx={self.local_expert_idx} on {local_dev}, "
            f"all other experts moved to CPU."
        )

    # ---------------------------------------------------------------------
    # profiler
    # ---------------------------------------------------------------------

    def _should_profile_iter(self, it: int) -> bool:
        return self.debug_profile and (self.debug_profile_start_iter <= it <= self.debug_profile_end_iter)

    @contextlib.contextmanager
    def _maybe_profile_iteration(self, it: int):
        if not self._should_profile_iter(it) or torch_profile is None:
            yield
            return

        acts = [ProfilerActivity.CPU]
        if self._is_cuda_device():
            acts.append(ProfilerActivity.CUDA)

        profile_dir = self.debug_profile_dir or os.path.join(os.getcwd(), "profile_traces")
        os.makedirs(profile_dir, exist_ok=True)

        with torch_profile(
            activities=acts,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            yield

        trace_path = os.path.join(profile_dir, f"rank{self.rank}_iter{it}.json")
        try:
            prof.export_chrome_trace(trace_path)
            log.info(f"[PROFILE][rank={self.rank}][it={it}] trace exported to {trace_path}")
        except Exception as e:
            log.warning(f"[PROFILE][rank={self.rank}][it={it}] export trace failed: {e}")

        try:
            sort_key = "self_cuda_time_total" if self._is_cuda_device() else "self_cpu_time_total"
            log.info("\n%s", prof.key_averages().table(sort_by=sort_key, row_limit=40))
        except Exception as e:
            log.warning(f"[PROFILE][rank={self.rank}][it={it}] print table failed: {e}")

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------

    def _warn_non_expert_trainables(self):
        expert_param_ids = {id(p) for expert in self.model.experts for p in expert.parameters()}
        outside = [
            name for name, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in expert_param_ids
        ]
        if outside:
            preview = outside[:10]
            suffix = "" if len(outside) <= 10 else f" ... (+{len(outside) - 10} more)"
            log.warning(
                "Found trainable params outside `model.experts`. "
                "Isolated optimizers will NOT update them: %s%s",
                preview, suffix
            )

    # ---------------------------------------------------------------------
    # batch prep
    # ---------------------------------------------------------------------

    def _prepare_expert_masks(self, batch_dict, range_dis, expert_idx):
        d_min, d_max = range_dis
        dist_edge = batch_dict['edge_lengths']

        if expert_idx == self.num_experts - 1:
            expert_edge_mask = (dist_edge >= d_min)
        else:
            expert_edge_mask = (dist_edge >= d_min) & (dist_edge < d_max)

        num_nodes = batch_dict["node_features"].shape[0]
        expert_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self._device_obj())
        if d_min > 0:
            expert_node_mask.fill_(False)

        return expert_edge_mask, expert_node_mask

    def _prepare_batch_bundle(self, batch, with_lengths=True):
        with self._tagger.tag("prepare_batch/to_device", it=self.iter):
            batch_dev = batch.to(self.device)

        batch_info = {
            "__slices__": batch_dev.__slices__,
            "__cumsum__": batch_dev.__cumsum__,
            "__cat_dims__": batch_dev.__cat_dims__,
            "__num_nodes_list__": batch_dev.__num_nodes_list__,
            "__data_class__": batch_dev.__data_class__,
        }

        with self._tagger.tag("prepare_batch/to_dict", it=self.iter):
            batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

        if with_lengths:
            with self._tagger.tag("prepare_batch/with_edge_vectors", it=self.iter):
                batch_dict = with_edge_vectors(batch_dict, with_lengths=True)

        return batch_dict, batch_info

    # -------------------- packed GPU tensor broadcast --------------------

    @staticmethod
    def _dtype_to_code(dtype: torch.dtype) -> str:
        mp = {
            torch.float32: "float32",
            torch.float64: "float64",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.int64: "int64",
            torch.int32: "int32",
            torch.int16: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        if dtype not in mp:
            raise TypeError(f"Unsupported dtype for packed broadcast: {dtype}")
        return mp[dtype]

    @staticmethod
    def _code_to_dtype(code: str) -> torch.dtype:
        mp = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        return mp[code]

    def _extract_batch_info_from_cpu_batch(self, batch):
        return {
            "__slices__": batch.__slices__,
            "__cumsum__": batch.__cumsum__,
            "__cat_dims__": batch.__cat_dims__,
            "__num_nodes_list__": batch.__num_nodes_list__,
            "__data_class__": batch.__data_class__,
        }

    def _split_tensor_and_object_items(self, d: Dict[str, Any]):
        tensor_items = []
        object_items = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                tensor_items.append((k, v))
            else:
                object_items[k] = v
        return tensor_items, object_items

    def _pack_tensor_groups(self, tensor_items: List[Tuple[str, torch.Tensor]]):
        groups = {}
        meta = []
        for k, t in tensor_items:
            t = t.contiguous()
            code = self._dtype_to_code(t.dtype)
            if code not in groups:
                groups[code] = []
            start = sum(x.numel() for x in groups[code])
            groups[code].append(t.reshape(-1))
            meta.append((k, code, tuple(t.shape), start, t.numel()))
        flat_groups = {}
        group_numel = {}
        for code, ts in groups.items():
            total = sum(x.numel() for x in ts)
            group_numel[code] = total
            if total == 0:
                flat_groups[code] = torch.empty((0,), dtype=self._code_to_dtype(code), device=self.device)
            else:
                flat_groups[code] = torch.cat(ts, dim=0)
        return meta, group_numel, flat_groups

    def _rebuild_tensor_groups_from_broadcast(self, schema, flat_groups):
        out = {}
        for k, code, shape, start, numel in schema["tensor_meta"]:
            dtype = self._code_to_dtype(code)
            if numel == 0:
                out[k] = torch.empty(shape, dtype=dtype, device=self.device)
            else:
                flat = flat_groups[code].narrow(0, int(start), int(numel))
                out[k] = flat.view(shape)
        out.update(schema["object_items"])
        return out

    def _broadcast_prepared_gpu_dict_packed(self, rank0_dict: Optional[Dict[str, Any]], tag_name: str):
        if not self._dist_ready():
            return rank0_dict

        schema_holder = [None]
        rank0_flat_groups = None

        if self.rank == 0:
            tensor_items, object_items = self._split_tensor_and_object_items(rank0_dict)
            tensor_meta, group_numel, rank0_flat_groups = self._pack_tensor_groups(tensor_items)
            schema_holder[0] = {
                "tensor_meta": tensor_meta,
                "group_numel": group_numel,
                "object_items": object_items,
            }

        with self._tagger.tag(f"{tag_name}/broadcast_schema", it=self.iter):
            dist.broadcast_object_list(schema_holder, src=0)

        schema = schema_holder[0]
        recv_flat_groups = {}
        for code, total_numel in schema["group_numel"].items():
            dtype = self._code_to_dtype(code)
            if self.rank == 0:
                flat = rank0_flat_groups[code]
            else:
                flat = torch.empty((int(total_numel),), dtype=dtype, device=self.device)

            if int(total_numel) > 0:
                with self._tagger.tag(
                    f"{tag_name}/broadcast_group",
                    it=self.iter,
                    extra=f"dtype={code} numel={int(total_numel)}"
                ):
                    dist.broadcast(flat, src=0)
            recv_flat_groups[code] = flat

        return self._rebuild_tensor_groups_from_broadcast(schema, recv_flat_groups)

    def _broadcast_prepared_bundle_rank0(
        self,
        batch,
        ref_batch=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self._dist_ready() or not self.distributed_rank0_prepare_batch:
            batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)
            ref_batch_dict = None
            ref_batch_info = None
            if ref_batch is not None:
                ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)
            return batch_dict, batch_info, ref_batch_dict, ref_batch_info

        batch_info_holder = [None]
        ref_batch_info_holder = [None]

        rank0_batch_dict = None
        rank0_ref_batch_dict = None

        if self.rank == 0:
            with self._tagger.tag("shared_batch/rank0_extract_batch_info", it=self.iter):
                batch_info_holder[0] = self._extract_batch_info_from_cpu_batch(batch)

            with self._tagger.tag("shared_batch/rank0_to_device", it=self.iter):
                batch_dev = batch.to(self.device)

            with self._tagger.tag("shared_batch/rank0_to_dict", it=self.iter):
                rank0_batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

            with self._tagger.tag("shared_batch/rank0_with_edge_vectors", it=self.iter):
                rank0_batch_dict = with_edge_vectors(rank0_batch_dict, with_lengths=True)

            if ref_batch is not None:
                with self._tagger.tag("shared_batch/rank0_ref_extract_batch_info", it=self.iter):
                    ref_batch_info_holder[0] = self._extract_batch_info_from_cpu_batch(ref_batch)

                with self._tagger.tag("shared_batch/rank0_ref_to_device", it=self.iter):
                    ref_batch_dev = ref_batch.to(self.device)

                with self._tagger.tag("shared_batch/rank0_ref_to_dict", it=self.iter):
                    rank0_ref_batch_dict = AtomicData.to_AtomicDataDict(ref_batch_dev)

                with self._tagger.tag("shared_batch/rank0_ref_with_edge_vectors", it=self.iter):
                    rank0_ref_batch_dict = with_edge_vectors(rank0_ref_batch_dict, with_lengths=True)

        with self._tagger.tag("shared_batch/broadcast_batch_info", it=self.iter):
            dist.broadcast_object_list(batch_info_holder, src=0)
            dist.broadcast_object_list(ref_batch_info_holder, src=0)

        batch_dict = self._broadcast_prepared_gpu_dict_packed(rank0_batch_dict, "shared_batch/main")
        batch_info = batch_info_holder[0]

        if ref_batch_info_holder[0] is not None:
            ref_batch_dict = self._broadcast_prepared_gpu_dict_packed(rank0_ref_batch_dict, "shared_batch/ref")
            ref_batch_info = ref_batch_info_holder[0]
        else:
            ref_batch_dict = None
            ref_batch_info = None

        return batch_dict, batch_info, ref_batch_dict, ref_batch_info

    # ---------------------------------------------------------------------
    # loss metric helpers
    # ---------------------------------------------------------------------

    def _resolve_loss_module(self, loss_obj):
        curr = loss_obj
        visited = set()
        while curr is not None and id(curr) not in visited:
            visited.add(id(curr))
            found_inner = None
            for attr in ("lossfunc", "loss_fn", "criterion", "method", "loss"):
                inner = getattr(curr, attr, None)
                if inner is None:
                    continue
                if isinstance(inner, nn.Module):
                    found_inner = inner
                    break
            if found_inner is None:
                break
            curr = found_inner
        return curr

    def _as_scalar_tensor(self, value, default=0.0, allow_none=False):
        if value is None:
            if allow_none:
                return None
            return torch.zeros((), dtype=self.dtype, device=self.device) + float(default)

        if torch.is_tensor(value):
            out = value.detach()
            if out.ndim != 0:
                out = out.mean()
            if out.device != self._device_obj():
                out = out.to(self.device)
            return out.to(dtype=self.dtype)

        return torch.tensor(float(value), dtype=self.dtype, device=self.device)

    def _to_float_scalar(self, value, default=0.0):
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            v = value.detach()
            if v.ndim != 0:
                v = v.mean()
            return float(v.item())
        return float(value)

    def _to_int_scalar(self, value, default=0):
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            v = value.detach()
            if v.ndim != 0:
                v = v.mean()
            return int(v.item())
        return int(value)

    def _snapshot_loss_metrics(self, loss_obj) -> Dict[str, Any]:
        loss_module = self._resolve_loss_module(loss_obj)

        out = {
            "onsite": self._as_scalar_tensor(getattr(loss_module, "last_onsite_loss", 0.0), default=0.0),
            "hopping": self._as_scalar_tensor(getattr(loss_module, "last_hopping_loss", 0.0), default=0.0),
            "z_loss": self._as_scalar_tensor(getattr(loss_module, "last_z_loss", None), allow_none=True),
            "expert_load_cv": self._as_scalar_tensor(getattr(loss_module, "expert_load_cv", None), allow_none=True),
        }

        for k in (
            "last_onsite_l1_sum", "last_onsite_mse_sum", "last_onsite_count",
            "last_hopping_l1_sum", "last_hopping_mse_sum", "last_hopping_count",
        ):
            v = getattr(loss_module, k, None)
            out[k] = self._as_scalar_tensor(v, default=0.0) if v is not None else None

        return out

    # ---------------------------------------------------------------------
    # core expert fwd/loss
    # ---------------------------------------------------------------------

    def _run_one_expert_loss(self, batch_dict, batch_info, criterion, expert_idx, range_dis, capture_metrics=False):
        with self._tagger.tag("expert/prepare_masks", it=self.iter, expert=expert_idx):
            expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)

        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask
        batch_copy["expert_idx"] = int(expert_idx)

        with self._tagger.tag("expert/model_forward", it=self.iter, expert=expert_idx):
            pred_batch = self.model(batch_copy)

        pred_batch["global_step"] = int(self.iter)

        with self._tagger.tag("expert/attach_batch_info", it=self.iter, expert=expert_idx):
            pred_batch.update(batch_info)
            batch_for_loss = batch_copy.copy()
            batch_for_loss.update(batch_info)

        with self._tagger.tag("expert/loss_forward", it=self.iter, expert=expert_idx):
            loss = criterion(pred_batch, batch_for_loss)

        out = {
            "loss": loss,
            "active_nodes": expert_node_mask.sum().detach(),
            "active_edges": expert_edge_mask.sum().detach(),
        }
        if capture_metrics:
            out.update(self._snapshot_loss_metrics(criterion))
        return out

    def _build_train_payload(
        self, batch_dict, batch_info, expert_idx, range_dis,
        ref_batch_dict=None, ref_batch_info=None, criterion=None
    ):
        if criterion is None:
            criterion = self.train_lossfunc

        main = self._run_one_expert_loss(
            batch_dict=batch_dict,
            batch_info=batch_info,
            criterion=criterion,
            expert_idx=expert_idx,
            range_dis=range_dis,
            capture_metrics=True
        )

        total_loss = main["loss"]
        active_nodes = main["active_nodes"]
        active_edges = main["active_edges"]

        onsite_weighted_sum = main["onsite"] * active_nodes.to(dtype=self.dtype)
        hopping_weighted_sum = main["hopping"] * active_edges.to(dtype=self.dtype)

        onsite_l1_sum = main["last_onsite_l1_sum"]
        onsite_mse_sum = main["last_onsite_mse_sum"]
        onsite_cnt = main["last_onsite_count"]
        hopping_l1_sum = main["last_hopping_l1_sum"]
        hopping_mse_sum = main["last_hopping_mse_sum"]
        hopping_cnt = main["last_hopping_count"]

        z_values = []
        load_cv_values = []
        if main["z_loss"] is not None:
            z_values.append(main["z_loss"])
        if main["expert_load_cv"] is not None:
            load_cv_values.append(main["expert_load_cv"])

        if ref_batch_dict is not None:
            ref_res = self._run_one_expert_loss(
                batch_dict=ref_batch_dict,
                batch_info=ref_batch_info,
                criterion=criterion,
                expert_idx=expert_idx,
                range_dis=range_dis,
                capture_metrics=True
            )

            total_loss = total_loss + ref_res["loss"]
            active_nodes = active_nodes + ref_res["active_nodes"]
            active_edges = active_edges + ref_res["active_edges"]
            onsite_weighted_sum = onsite_weighted_sum + ref_res["onsite"] * ref_res["active_nodes"].to(dtype=self.dtype)
            hopping_weighted_sum = hopping_weighted_sum + ref_res["hopping"] * ref_res["active_edges"].to(dtype=self.dtype)

            if onsite_l1_sum is not None and ref_res["last_onsite_l1_sum"] is not None:
                onsite_l1_sum = onsite_l1_sum + ref_res["last_onsite_l1_sum"]
                onsite_mse_sum = onsite_mse_sum + ref_res["last_onsite_mse_sum"]
                onsite_cnt = onsite_cnt + ref_res["last_onsite_count"]
            if hopping_l1_sum is not None and ref_res["last_hopping_l1_sum"] is not None:
                hopping_l1_sum = hopping_l1_sum + ref_res["last_hopping_l1_sum"]
                hopping_mse_sum = hopping_mse_sum + ref_res["last_hopping_mse_sum"]
                hopping_cnt = hopping_cnt + ref_res["last_hopping_count"]

            if ref_res["z_loss"] is not None:
                z_values.append(ref_res["z_loss"])
            if ref_res["expert_load_cv"] is not None:
                load_cv_values.append(ref_res["expert_load_cv"])

        active_nodes_safe = active_nodes.to(dtype=self.dtype).clamp_min(1.0)
        active_edges_safe = active_edges.to(dtype=self.dtype).clamp_min(1.0)
        expert_onsite = onsite_weighted_sum / active_nodes_safe
        expert_hopping = hopping_weighted_sum / active_edges_safe

        return {
            "loss": total_loss,
            "expert_onsite": expert_onsite.detach(),
            "expert_hopping": expert_hopping.detach(),
            "onsite_weighted_sum": onsite_weighted_sum.detach(),
            "hopping_weighted_sum": hopping_weighted_sum.detach(),
            "active_nodes": active_nodes.detach(),
            "active_edges": active_edges.detach(),
            "onsite_l1_sum": onsite_l1_sum.detach() if torch.is_tensor(onsite_l1_sum) else None,
            "onsite_mse_sum": onsite_mse_sum.detach() if torch.is_tensor(onsite_mse_sum) else None,
            "onsite_cnt": onsite_cnt.detach() if torch.is_tensor(onsite_cnt) else None,
            "hopping_l1_sum": hopping_l1_sum.detach() if torch.is_tensor(hopping_l1_sum) else None,
            "hopping_mse_sum": hopping_mse_sum.detach() if torch.is_tensor(hopping_mse_sum) else None,
            "hopping_cnt": hopping_cnt.detach() if torch.is_tensor(hopping_cnt) else None,
            "z_values": [z.detach() for z in z_values],
            "load_cv_values": [cv.detach() for cv in load_cv_values],
        }

    # ---------------------------------------------------------------------
    # stitched loss helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _maybe_call_or_value(x, default: float = 1.0) -> float:
        if x is None:
            return float(default)
        if callable(x):
            try:
                return float(x())
            except Exception:
                return float(default)
        try:
            return float(x)
        except Exception:
            return float(default)

    def _compute_stitched_loss_by_reduce(self, payloads: List[Dict[str, Any]], criterion=None) -> Optional[torch.Tensor]:
        if criterion is None:
            criterion = self.train_lossfunc

        if (not self.log_single_model_compatible_loss) or (self.log_single_model_compatible_loss_mode != "reduce"):
            return None

        onsite_l1_sum = None
        onsite_mse_sum = None
        onsite_cnt = None
        hopping_l1_sum = None
        hopping_mse_sum = None
        hopping_cnt = None
        z_vals = []

        for p in payloads:
            if p is None:
                continue

            if p.get("onsite_l1_sum") is not None:
                onsite_l1_sum = p["onsite_l1_sum"] if onsite_l1_sum is None else (onsite_l1_sum + p["onsite_l1_sum"])
                onsite_mse_sum = p["onsite_mse_sum"] if onsite_mse_sum is None else (onsite_mse_sum + p["onsite_mse_sum"])
                onsite_cnt = p["onsite_cnt"] if onsite_cnt is None else (onsite_cnt + p["onsite_cnt"])

            if p.get("hopping_l1_sum") is not None:
                hopping_l1_sum = p["hopping_l1_sum"] if hopping_l1_sum is None else (hopping_l1_sum + p["hopping_l1_sum"])
                hopping_mse_sum = p["hopping_mse_sum"] if hopping_mse_sum is None else (hopping_mse_sum + p["hopping_mse_sum"])
                hopping_cnt = p["hopping_cnt"] if hopping_cnt is None else (hopping_cnt + p["hopping_cnt"])

            for z in p.get("z_values", []):
                if z is not None:
                    z_vals.append(z)

        if onsite_cnt is None and hopping_cnt is None:
            return None

        def _safe_mean(sum_t, cnt_t):
            if sum_t is None or cnt_t is None:
                return torch.zeros((), dtype=self.dtype, device=self.device)
            return sum_t / cnt_t.to(dtype=self.dtype).clamp_min(1.0)

        onsite_l1_mean = _safe_mean(onsite_l1_sum, onsite_cnt)
        onsite_mse_mean = _safe_mean(onsite_mse_sum, onsite_cnt)
        hopping_l1_mean = _safe_mean(hopping_l1_sum, hopping_cnt)
        hopping_mse_mean = _safe_mean(hopping_mse_sum, hopping_cnt)

        onsite_loss = 0.5 * (onsite_l1_mean + torch.sqrt(onsite_mse_mean))
        hopping_loss = 0.5 * (hopping_l1_mean + torch.sqrt(hopping_mse_mean))

        loss_module = self._resolve_loss_module(criterion)
        onsite_boost = bool(getattr(loss_module, "onsite_boost", False))
        onsite_boost_w = self._maybe_call_or_value(getattr(loss_module, "_current_onsite_weight", None), default=1.0)
        z_coef = float(getattr(loss_module, "z_loss_coef", 0.0))

        if onsite_boost:
            total = onsite_boost_w * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        if z_coef > 0.0 and len(z_vals) > 0:
            z_mean = torch.stack([z.to(self.device, dtype=self.dtype) for z in z_vals]).mean()
            total = total + z_coef * z_mean

        return total.detach()

    def _compute_compatible_loss_from_pack(self, pack: torch.Tensor, criterion=None):
        if criterion is None:
            criterion = self.train_lossfunc

        if (not self.log_single_model_compatible_loss) or (self.log_single_model_compatible_loss_mode != "reduce"):
            return None

        onsite_cnt = pack[self._P_ONSITE_CNT_SUM]
        hopping_cnt = pack[self._P_HOPPING_CNT_SUM]

        if float(onsite_cnt.item()) <= 0.0 and float(hopping_cnt.item()) <= 0.0:
            return None

        def _safe_mean(sum_t, cnt_t):
            return sum_t / cnt_t.to(dtype=self.dtype).clamp_min(1.0)

        onsite_l1_mean = _safe_mean(pack[self._P_ONSITE_L1_SUM], onsite_cnt)
        onsite_mse_mean = _safe_mean(pack[self._P_ONSITE_MSE_SUM], onsite_cnt)
        hopping_l1_mean = _safe_mean(pack[self._P_HOPPING_L1_SUM], hopping_cnt)
        hopping_mse_mean = _safe_mean(pack[self._P_HOPPING_MSE_SUM], hopping_cnt)

        onsite_loss = 0.5 * (onsite_l1_mean + torch.sqrt(onsite_mse_mean))
        hopping_loss = 0.5 * (hopping_l1_mean + torch.sqrt(hopping_mse_mean))

        loss_module = self._resolve_loss_module(criterion)
        onsite_boost = bool(getattr(loss_module, "onsite_boost", False))
        onsite_boost_w = self._maybe_call_or_value(getattr(loss_module, "_current_onsite_weight", None), default=1.0)
        z_coef = float(getattr(loss_module, "z_loss_coef", 0.0))

        if onsite_boost:
            total = onsite_boost_w * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        if z_coef > 0.0 and float(pack[self._P_Z_CNT].item()) > 0.0:
            total = total + z_coef * (pack[self._P_Z_SUM] / pack[self._P_Z_CNT].clamp_min(1.0))

        return total.detach()

    def _compute_local_compatible_loss_from_payload(self, payload: Dict[str, Any], criterion=None) -> torch.Tensor:
        out = self._compute_stitched_loss_by_reduce([payload], criterion=criterion)
        if out is None:
            out = payload["loss_detached"].detach()
        return out.detach()

    # ---------------------------------------------------------------------
    # display window buffers
    # ---------------------------------------------------------------------

    def _reset_display_window_buffers(self):
        dev = self._device_obj()
        self._display_window_pack_local = torch.zeros((self._PACK_LEN,), dtype=self.dtype, device=dev)
        self._display_window_expert_onsite_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_hopping_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_active_nodes_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_expert_active_edges_sum_local = torch.zeros((), dtype=self.dtype, device=dev)
        self._display_window_last_lr_local = 0.0

    def _has_pending_display_window(self) -> bool:
        return float(self._display_window_pack_local[self._P_STEP_COUNT].item()) > 0.0

    def _make_step_pack(self, payload: Dict[str, Any]) -> torch.Tensor:
        vec = torch.zeros((self._PACK_LEN,), dtype=self.dtype, device=self.device)

        vec[self._P_LOSS_OPT_SUM] = self._as_scalar_tensor(payload.get("loss_detached", 0.0))
        vec[self._P_ONSITE_WEIGHTED_SUM] = self._as_scalar_tensor(payload.get("onsite_weighted_sum", 0.0))
        vec[self._P_HOPPING_WEIGHTED_SUM] = self._as_scalar_tensor(payload.get("hopping_weighted_sum", 0.0))
        vec[self._P_ACTIVE_NODES_SUM] = self._as_scalar_tensor(payload.get("active_nodes", 0.0))
        vec[self._P_ACTIVE_EDGES_SUM] = self._as_scalar_tensor(payload.get("active_edges", 0.0))

        vec[self._P_ONSITE_L1_SUM] = self._as_scalar_tensor(payload.get("onsite_l1_sum", 0.0))
        vec[self._P_ONSITE_MSE_SUM] = self._as_scalar_tensor(payload.get("onsite_mse_sum", 0.0))
        vec[self._P_ONSITE_CNT_SUM] = self._as_scalar_tensor(payload.get("onsite_cnt", 0.0))
        vec[self._P_HOPPING_L1_SUM] = self._as_scalar_tensor(payload.get("hopping_l1_sum", 0.0))
        vec[self._P_HOPPING_MSE_SUM] = self._as_scalar_tensor(payload.get("hopping_mse_sum", 0.0))
        vec[self._P_HOPPING_CNT_SUM] = self._as_scalar_tensor(payload.get("hopping_cnt", 0.0))

        z_vals = [self._as_scalar_tensor(z, default=0.0) for z in payload.get("z_values", []) if z is not None]
        if z_vals:
            vec[self._P_Z_SUM] = torch.stack(z_vals).sum()
            vec[self._P_Z_CNT] = float(len(z_vals))

        cv_vals = [self._as_scalar_tensor(v, default=0.0) for v in payload.get("load_cv_values", []) if v is not None]
        if cv_vals:
            vec[self._P_CV_SUM] = torch.stack(cv_vals).sum()
            vec[self._P_CV_CNT] = float(len(cv_vals))

        vec[self._P_GRAD_NORM_SUM] = self._as_scalar_tensor(payload.get("grad_norm", 0.0))
        vec[self._P_STEP_COUNT] = 1.0
        return vec

    def _update_display_window_local(self, payload: Dict[str, Any], current_local_lr: float):
        self._display_window_pack_local += self._make_step_pack(payload)
        self._display_window_expert_onsite_sum_local += self._as_scalar_tensor(payload["expert_onsite"], default=0.0)
        self._display_window_expert_hopping_sum_local += self._as_scalar_tensor(payload["expert_hopping"], default=0.0)
        self._display_window_expert_active_nodes_sum_local += self._as_scalar_tensor(payload["active_nodes"], default=0.0)
        self._display_window_expert_active_edges_sum_local += self._as_scalar_tensor(payload["active_edges"], default=0.0)
        self._display_window_last_lr_local = float(current_local_lr)

    def _should_flush_display_window_now(self, it: int) -> bool:
        return (it == 1) or (it % self.display_sync_freq == 0)

    def _gather_display_window_expert_metrics(self) -> List[torch.Tensor]:
        local_steps = max(float(self._display_window_pack_local[self._P_STEP_COUNT].item()), 1.0)

        local_metric = torch.stack([
            self._display_window_expert_onsite_sum_local / local_steps,
            self._display_window_expert_hopping_sum_local / local_steps,
            self._display_window_pack_local[self._P_GRAD_NORM_SUM] / local_steps,
            torch.tensor(float(self._display_window_last_lr_local), dtype=self.dtype, device=self.device),
            self._display_window_expert_active_nodes_sum_local / local_steps,
            self._display_window_expert_active_edges_sum_local / local_steps,
        ])

        if not self._dist_ready():
            return [local_metric]

        gathered = [torch.zeros_like(local_metric) for _ in range(self.world_size)]
        self._all_gather_(gathered, local_metric, name="dist/all_gather(display_window_expert_metrics)")
        return gathered

    def _flush_display_window(self, time_idx: int) -> Optional[Dict[str, Any]]:
        if not self._has_pending_display_window():
            return None

        with self._tagger.tag("display/window_reduce", it=time_idx, extra=f"freq={self.display_sync_freq}"):
            reduced_pack = self._display_window_pack_local.clone()
            self._all_reduce_(reduced_pack, name="dist/all_reduce(display_window_metrics_packed)")
            gathered = self._gather_display_window_expert_metrics()

        world = self.world_size if self._dist_ready() else 1
        total_steps = max(float(reduced_pack[self._P_STEP_COUNT].item()) / world, 1.0)

        train_loss_opt_mean = reduced_pack[self._P_LOSS_OPT_SUM] / total_steps
        compatible_train_loss = self._compute_compatible_loss_from_pack(reduced_pack, self.train_lossfunc)
        train_loss_show = compatible_train_loss if compatible_train_loss is not None else train_loss_opt_mean

        global_onsite = reduced_pack[self._P_ONSITE_WEIGHTED_SUM] / reduced_pack[self._P_ACTIVE_NODES_SUM].clamp_min(1.0)
        global_hopping = reduced_pack[self._P_HOPPING_WEIGHTED_SUM] / reduced_pack[self._P_ACTIVE_EDGES_SUM].clamp_min(1.0)
        total_grad_norm_mean = reduced_pack[self._P_GRAD_NORM_SUM] / reduced_pack[self._P_STEP_COUNT].clamp_min(1.0)

        avg_lr = sum(float(vec[3].item()) for vec in gathered) / max(len(gathered), 1)

        state = {
            'field': 'iteration',
            'window_steps': int(round(total_steps)),
            "train_loss": train_loss_show.detach() if torch.is_tensor(train_loss_show) else torch.tensor(float(train_loss_show), device=self.device, dtype=self.dtype),
            "train_loss_opt": train_loss_opt_mean.detach() if torch.is_tensor(train_loss_opt_mean) else torch.tensor(float(train_loss_opt_mean), device=self.device, dtype=self.dtype),
            "lr": float(avg_lr),
            "total_grad_norm": float(total_grad_norm_mean.item()),
            "train_onsite_loss": float(global_onsite.item()),
            "train_hopping_loss": float(global_hopping.item()),
        }

        for expert_idx, vec in enumerate(gathered):
            state[f"expert_{expert_idx}_onsite"] = float(vec[0].item())
            state[f"expert_{expert_idx}_hopping"] = float(vec[1].item())
            state[f"expert_{expert_idx}_lr"] = float(vec[3].item())
            state[f"expert_{expert_idx}_active_nodes"] = float(vec[4].item())
            state[f"expert_{expert_idx}_active_edges"] = float(vec[5].item())

        if float(reduced_pack[self._P_CV_CNT].item()) > 0.0:
            state["expert_load_cv"] = float((reduced_pack[self._P_CV_SUM] / reduced_pack[self._P_CV_CNT]).item())
        if float(reduced_pack[self._P_Z_CNT].item()) > 0.0:
            state["mean_max_prob"] = float((reduced_pack[self._P_Z_SUM] / reduced_pack[self._P_Z_CNT]).item())

        self._reset_display_window_buffers()
        return state

    # ---------------------------------------------------------------------
    # scheduler
    # ---------------------------------------------------------------------

    def _get_epoch_scheduler_metric(self):
        validation_stat = self.stats.get("validation_loss", {})
        if isinstance(validation_stat, dict) and ("epoch_mean" in validation_stat):
            metric = validation_stat["epoch_mean"]
        else:
            train_stat = self.stats.get("train_loss", {})
            metric = train_stat.get("epoch_mean", None) if isinstance(train_stat, dict) else None

        if torch.is_tensor(metric):
            metric = metric.detach()
            if metric.ndim != 0:
                metric = metric.mean()
            return float(metric.item())
        return metric

    def _local_scheduler_step(self, metric_tensor: torch.Tensor):
        if not self.update_lr_per_iter:
            return

        if torch.is_tensor(metric_tensor):
            m = metric_tensor.detach()
            if m.ndim != 0:
                m = m.mean()
            metric_float = float(m.item())
        else:
            metric_float = float(metric_tensor)

        if self.distributed_expert:
            sch = self.lr_schedulers[self.local_expert_idx]
            if sch is None:
                return

            with self._tagger.tag("scheduler/local_step", it=self.iter, expert=self.local_expert_idx, extra=f"metric={metric_float:.6g}"):
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.iter > 1:
                        sch.step(metric_float)
                else:
                    sch.step()
        else:
            with self._tagger.tag("scheduler/local_step(all)", it=self.iter, extra=f"metric={metric_float:.6g}"):
                for sch in self.lr_schedulers:
                    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if self.iter > 1:
                            sch.step(metric_float)
                    else:
                        sch.step()

    def _step_epoch_schedulers(self):
        metric = self._get_epoch_scheduler_metric()
        metric_float = None if metric is None else self._to_float_scalar(metric)

        def _step_one_scheduler(sch, expert_idx=None):
            if sch is None:
                return

            extra = f"metric={metric_float:.6g}" if metric_float is not None else "metric=None"
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metric_float is None:
                    log.warning("Skip epoch LR scheduler step: no epoch metric is available.")
                    return
                with self._tagger.tag("scheduler/epoch_step", it=self.iter, expert=expert_idx, extra=extra):
                    sch.step(metric_float)
            else:
                with self._tagger.tag("scheduler/epoch_step", it=self.iter, expert=expert_idx, extra=extra):
                    sch.step()

        if self.distributed_expert:
            _step_one_scheduler(self.lr_schedulers[self.local_expert_idx], expert_idx=self.local_expert_idx)
        else:
            for expert_idx, sch in enumerate(self.lr_schedulers):
                _step_one_scheduler(sch, expert_idx=expert_idx)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for i in range(self.ep, epochs + 1):
            self.epoch()
            self.call_plugins(queue_name='epoch', time=i)

            if not self.update_lr_per_iter:
                self._step_epoch_schedulers()

            self.update()
            self.ep += 1

    # ---------------------------------------------------------------------
    # distributed expert iteration
    # ---------------------------------------------------------------------

    def _iteration_distributed_expert_prepared(
        self,
        batch_dict,
        batch_info,
        ref_batch_dict=None,
        ref_batch_info=None
    ):
        with self._tagger.tag("iteration/entry", it=self.iter):
            self.model.train()

        local_idx = self.local_expert_idx
        local_opt = self.optimizers[local_idx]

        with self._tagger.tag("iteration/zero_grad(local)", it=self.iter, expert=local_idx):
            local_opt.zero_grad(set_to_none=True)

        with self._tagger.tag("expert/build_payload(fwd+loss)", it=self.iter, expert=local_idx):
            payload = self._build_train_payload(
                batch_dict=batch_dict,
                batch_info=batch_info,
                expert_idx=local_idx,
                range_dis=self.distance_ranges[local_idx],
                ref_batch_dict=ref_batch_dict,
                ref_batch_info=ref_batch_info,
                criterion=self.train_lossfunc,
            )

        loss_local = payload["loss"]

        with self._tagger.tag("expert/backward", it=self.iter, expert=local_idx):
            loss_local.backward()

        with self._tagger.tag("expert/clip_grad_norm", it=self.iter, expert=local_idx):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.experts[local_idx].parameters(),
                max_norm=self.clip_grad_norm
            )

        with self._tagger.tag("expert/optimizer_step", it=self.iter, expert=local_idx):
            local_opt.step()

        payload["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(
            float(grad_norm), device=self.device, dtype=self.dtype
        )
        payload["loss_detached"] = loss_local.detach()
        del payload["loss"]

        with self._tagger.tag("iteration/compute_local_train_loss_compatible", it=self.iter):
            local_sched_metric = self._compute_local_compatible_loss_from_payload(payload, self.train_lossfunc)

        self._local_scheduler_step(local_sched_metric)

        current_local_lr = float(local_opt.param_groups[0]['lr'])
        self._update_display_window_local(payload, current_local_lr)

        if self._should_flush_display_window_now(self.iter):
            state = self._flush_display_window(time_idx=self.iter)
            if state is not None:
                with self._tagger.tag("iteration/call_plugins", it=self.iter):
                    self.call_plugins(queue_name='iteration', time=self.iter, **state)

        with self._tagger.tag("iteration/exit", it=self.iter):
            self.iter += 1

        return loss_local.detach()

    def _iteration_distributed_expert(self, batch, ref_batch=None):
        with self._tagger.tag("iteration/prepare_batch", it=self.iter):
            batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

        ref_batch_dict = None
        ref_batch_info = None
        if ref_batch is not None:
            with self._tagger.tag("iteration/prepare_ref_batch", it=self.iter):
                ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)

        return self._iteration_distributed_expert_prepared(
            batch_dict=batch_dict,
            batch_info=batch_info,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info
        )

    def _iteration_distributed_expert_shared(self, batch, ref_batch=None):
        with self._tagger.tag("iteration/prepare_batch(shared_rank0_only)", it=self.iter):
            batch_dict, batch_info, ref_batch_dict, ref_batch_info = self._broadcast_prepared_bundle_rank0(
                batch=batch,
                ref_batch=ref_batch
            )

        return self._iteration_distributed_expert_prepared(
            batch_dict=batch_dict,
            batch_info=batch_info,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info
        )

    # ---------------------------------------------------------------------
    # public iteration
    # ---------------------------------------------------------------------

    def iteration(self, batch, ref_batch=None):
        t_now = time.perf_counter()
        if self._t_last_iter_end is not None and self.debug_tags and (self.iter % self.debug_tag_freq == 0):
            log.info(f"[TAG][it={self.iter}][data_wait(outside_iteration)] dt={(t_now - self._t_last_iter_end):.4f}s")

        try:
            with self._maybe_profile_iteration(self.iter):
                if self.distributed_expert:
                    if self.distributed_rank0_prepare_batch:
                        return self._iteration_distributed_expert_shared(batch, ref_batch=ref_batch)
                    return self._iteration_distributed_expert(batch, ref_batch=ref_batch)

                # single-process fallback
                with self._tagger.tag("iteration/entry", it=self.iter):
                    self.model.train()

                with self._tagger.tag("iteration/prepare_batch", it=self.iter):
                    batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

                ref_batch_dict = None
                ref_batch_info = None
                if ref_batch is not None:
                    with self._tagger.tag("iteration/prepare_ref_batch", it=self.iter):
                        ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)

                total_loss_opt = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
                expert_grad_norms = []

                global_onsite_sum = 0.0
                global_hopping_sum = 0.0
                total_active_nodes = 0
                total_active_edges = 0
                expert_onsite_dict = {}
                expert_hopping_dict = {}
                z_metric_values = []
                expert_load_cv_values = []

                reduce_payloads: List[Dict[str, Any]] = []

                def collect_payload(expert_idx, payload):
                    nonlocal total_loss_opt
                    nonlocal global_onsite_sum, global_hopping_sum
                    nonlocal total_active_nodes, total_active_edges

                    total_loss_opt = total_loss_opt + payload["loss_detached"]
                    expert_grad_norms.append(self._to_float_scalar(payload["grad_norm"]))

                    expert_onsite = self._to_float_scalar(payload["expert_onsite"])
                    expert_hopping = self._to_float_scalar(payload["expert_hopping"])
                    expert_onsite_dict[f"expert_{expert_idx}_onsite"] = expert_onsite
                    expert_hopping_dict[f"expert_{expert_idx}_hopping"] = expert_hopping

                    global_onsite_sum += self._to_float_scalar(payload["onsite_weighted_sum"])
                    global_hopping_sum += self._to_float_scalar(payload["hopping_weighted_sum"])
                    total_active_nodes += self._to_int_scalar(payload["active_nodes"])
                    total_active_edges += self._to_int_scalar(payload["active_edges"])

                    for z in payload.get("z_values", []):
                        if z is not None:
                            z_metric_values.append(self._to_float_scalar(z))
                    for cv in payload.get("load_cv_values", []):
                        if cv is not None:
                            expert_load_cv_values.append(self._to_float_scalar(cv))

                    reduce_payloads.append(payload)

                with self._tagger.tag("iteration/zero_grad(all)", it=self.iter):
                    for opt in self.optimizers:
                        opt.zero_grad(set_to_none=True)

                payload_list = []
                for expert_idx, range_dis in enumerate(self.distance_ranges):
                    with self._tagger.tag("expert/build_payload(fwd+loss)", it=self.iter, expert=expert_idx):
                        payload = self._build_train_payload(
                            batch_dict=batch_dict,
                            batch_info=batch_info,
                            expert_idx=expert_idx,
                            range_dis=range_dis,
                            ref_batch_dict=ref_batch_dict,
                            ref_batch_info=ref_batch_info,
                        )

                    loss_expert = payload["loss"]

                    with self._tagger.tag("expert/backward", it=self.iter, expert=expert_idx):
                        loss_expert.backward()

                    with self._tagger.tag("expert/clip_grad_norm", it=self.iter, expert=expert_idx):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.experts[expert_idx].parameters(),
                            max_norm=self.clip_grad_norm
                        )

                    with self._tagger.tag("expert/optimizer_step", it=self.iter, expert=expert_idx):
                        self.optimizers[expert_idx].step()

                    payload["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(
                        float(grad_norm), device=self.device, dtype=self.dtype
                    )
                    payload["loss_detached"] = loss_expert.detach()
                    del payload["loss"]
                    payload_list.append(payload)

                with self._tagger.tag("iteration/collect_payloads", it=self.iter):
                    for expert_idx, payload in enumerate(payload_list):
                        collect_payload(expert_idx, payload)

                global_onsite = global_onsite_sum / max(total_active_nodes, 1)
                global_hopping = global_hopping_sum / max(total_active_edges, 1)

                with self._tagger.tag("iteration/compute_train_loss_compatible(reduce)", it=self.iter):
                    comparable_train_loss = self._compute_stitched_loss_by_reduce(reduce_payloads, self.train_lossfunc)

                final_train_loss = comparable_train_loss if comparable_train_loss is not None else total_loss_opt
                self._local_scheduler_step(final_train_loss)

                # ---------------- NEW: make lr consistent: use mean lr across experts ----------------
                avg_lr = sum(float(opt.param_groups[0]['lr']) for opt in self.optimizers) / max(len(self.optimizers), 1)
                # ----------------------------------------------------------------------------------

                state = {
                    'field': 'iteration',
                    'window_steps': 1,
                    "train_loss": final_train_loss,
                    "train_loss_opt": total_loss_opt,
                    "lr": avg_lr,
                    "total_grad_norm": sum(expert_grad_norms) / max(len(expert_grad_norms), 1),
                    "train_onsite_loss": global_onsite,
                    "train_hopping_loss": global_hopping,
                }

                for i in range(self.num_experts):
                    state[f"expert_{i}_onsite"] = expert_onsite_dict.get(f"expert_{i}_onsite", 0.0)
                    state[f"expert_{i}_hopping"] = expert_hopping_dict.get(f"expert_{i}_hopping", 0.0)
                    state[f"expert_{i}_lr"] = float(self.optimizers[i].param_groups[0]['lr'])

                if expert_load_cv_values:
                    state["expert_load_cv"] = sum(expert_load_cv_values) / len(expert_load_cv_values)
                if z_metric_values:
                    state["mean_max_prob"] = sum(z_metric_values) / len(z_metric_values)

                with self._tagger.tag("iteration/call_plugins", it=self.iter):
                    self.call_plugins(queue_name='iteration', time=self.iter, **state)

                with self._tagger.tag("iteration/exit", it=self.iter):
                    self.iter += 1

                return total_loss_opt

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._tagger.dump_cuda_mem_summary(where="iteration() top-level")
            raise
        finally:
            self._t_last_iter_end = time.perf_counter()

    # ---------------------------------------------------------------------
    # epoch override
    # ---------------------------------------------------------------------

    def epoch(self) -> None:
        if self.distributed_expert and self.distributed_rank0_prepare_batch:
            train_iter = iter(self.train_loader) if self.rank == 0 else None
            ref_iter = iter(self.reference_loader) if (self.use_reference and self.rank == 0) else None

            n_step = len(self.train_loader)
            for _ in range(n_step):
                if self.rank == 0:
                    batch = next(train_iter)
                    if self.use_reference:
                        try:
                            ref_batch = next(ref_iter)
                        except StopIteration:
                            ref_iter = iter(self.reference_loader)
                            ref_batch = next(ref_iter)
                    else:
                        ref_batch = None
                else:
                    batch = None
                    ref_batch = None

                self.iteration(batch, ref_batch)

            if self._has_pending_display_window():
                flush_time = max(self.iter - 1, 1)
                state = self._flush_display_window(time_idx=flush_time)
                if state is not None:
                    self.call_plugins(queue_name='iteration', time=flush_time, **state)
            return

        if self.use_reference:
            ref_iter = iter(self.reference_loader)
            for ibatch in self.train_loader:
                try:
                    ref_batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(self.reference_loader)
                    ref_batch = next(ref_iter)
                self.iteration(ibatch, ref_batch)
        else:
            for ibatch in self.train_loader:
                self.iteration(ibatch)

        if self.distributed_expert and self._has_pending_display_window():
            flush_time = max(self.iter - 1, 1)
            state = self._flush_display_window(time_idx=flush_time)
            if state is not None:
                self.call_plugins(queue_name='iteration', time=flush_time, **state)

    # ---------------------------------------------------------------------
    # validation
    # ---------------------------------------------------------------------

    def _run_full_batch_loss(self, batch_dict, batch_info, criterion):
        batch_copy = batch_dict.copy()
        batch_for_loss = batch_copy.copy()

        pred_batch = self.model(batch_copy)
        pred_batch["global_step"] = int(self.iter)
        pred_batch.update(batch_info)
        batch_for_loss.update(batch_info)

        return criterion(pred_batch, batch_for_loss)

    def validation(self, fast=True):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            self.model.eval()

            for batch in self.validation_loader:
                with self._tagger.tag("validation/prepare_batch", it=self.iter):
                    batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

                if self.distributed_expert:
                    local_idx = self.local_expert_idx
                    payload = self._build_train_payload(
                        batch_dict=batch_dict,
                        batch_info=batch_info,
                        expert_idx=local_idx,
                        range_dis=self.distance_ranges[local_idx],
                        ref_batch_dict=None,
                        ref_batch_info=None,
                        criterion=self.validation_lossfunc,
                    )

                    payload["loss_detached"] = payload["loss"].detach()

                    with self._tagger.tag("validation/reduce_packed_metrics_dist", it=self.iter):
                        reduced_pack = self._make_step_pack(payload)
                        self._all_reduce_(reduced_pack, name="dist/all_reduce(validation_metrics_packed)")

                    if self.log_single_model_compatible_loss and self.log_single_model_compatible_loss_mode == "reduce":
                        with self._tagger.tag("validation/compute_reduce_loss_dist_packed", it=self.iter):
                            loss_i = self._compute_compatible_loss_from_pack(reduced_pack, self.validation_lossfunc)
                        if loss_i is None:
                            loss_i = reduced_pack[self._P_LOSS_OPT_SUM].detach() / max(self.world_size, 1)
                    else:
                        loss_i = reduced_pack[self._P_LOSS_OPT_SUM].detach() / max(self.world_size, 1)

                else:
                    if self.log_single_model_compatible_loss and self.log_single_model_compatible_loss_mode == "reduce":
                        payloads = []
                        for expert_idx, range_dis in enumerate(self.distance_ranges):
                            res = self._run_one_expert_loss(
                                batch_dict=batch_dict,
                                batch_info=batch_info,
                                criterion=self.validation_lossfunc,
                                expert_idx=expert_idx,
                                range_dis=range_dis,
                                capture_metrics=True
                            )
                            payloads.append({
                                "onsite_l1_sum": res.get("last_onsite_l1_sum", None),
                                "onsite_mse_sum": res.get("last_onsite_mse_sum", None),
                                "onsite_cnt": res.get("last_onsite_count", None),
                                "hopping_l1_sum": res.get("last_hopping_l1_sum", None),
                                "hopping_mse_sum": res.get("last_hopping_mse_sum", None),
                                "hopping_cnt": res.get("last_hopping_count", None),
                                "z_values": [res["z_loss"]] if res.get("z_loss", None) is not None else [],
                            })

                        with self._tagger.tag("validation/compute_reduce_loss", it=self.iter):
                            loss_i = self._compute_stitched_loss_by_reduce(payloads, self.validation_lossfunc)

                        if loss_i is None:
                            with self._tagger.tag("validation/fallback_full_forward", it=self.iter):
                                loss_i = self._run_full_batch_loss(batch_dict, batch_info, self.validation_lossfunc)
                    else:
                        with self._tagger.tag("validation/full_forward_stitched", it=self.iter):
                            loss_i = self._run_full_batch_loss(batch_dict, batch_info, self.validation_lossfunc)

                total_loss = total_loss + loss_i.detach()

                if fast:
                    break

        if not fast:
            total_loss = total_loss / len(self.validation_loader)

        return total_loss

    # ---------------------------------------------------------------------
    # restart
    # ---------------------------------------------------------------------

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options={}, common_options={}, reference_datasets=None,
                validation_datasets=None, distributed_expert=False, rank=0, world_size=1):
        map_loc = "cpu" if distributed_expert else (
            common_options["device"] if len(common_options) > 0 and "device" in common_options else "cpu"
        )
        ckpt = torch.load(checkpoint, map_location=map_loc, weights_only=False)

        merged_train_options = copy.deepcopy(ckpt["config"].get("train_options", {}))
        merged_train_options.update(train_options or {})

        merged_common_options = copy.deepcopy(ckpt["config"]["common_options"])
        merged_common_options.update(common_options or {})

        build_common_options = copy.deepcopy(merged_common_options)
        if distributed_expert:
            build_common_options["device"] = "cpu"

        model = build_model(
            checkpoint=checkpoint,
            model_options=ckpt["config"]["model_options"],
            common_options=build_common_options,
            train_options=merged_train_options
        )

        distance_ranges = merged_train_options.get(
            "distance_ranges",
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
        )

        trainer = cls(
            distance_ranges=distance_ranges,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
            train_options=merged_train_options,
            common_options=merged_common_options,
            distributed_expert=distributed_expert,
            rank=rank,
            world_size=world_size,
        )

        trainer.ep = ckpt["epoch"] + 1
        trainer.iter = ckpt["iteration"] + 1
        trainer.stats = ckpt["stats"]

        if distributed_expert:
            idx = trainer.local_expert_idx
            opt_states = ckpt.get("optimizers_state_dict", None)
            sch_states = ckpt.get("lr_schedulers_state_dict", None)
            if opt_states is not None and trainer.optimizers[idx] is not None:
                trainer.optimizers[idx].load_state_dict(opt_states[idx])
            if sch_states is not None and trainer.lr_schedulers[idx] is not None:
                trainer.lr_schedulers[idx].load_state_dict(sch_states[idx])
        else:
            for key in cls.object_keys:
                items = getattr(trainer, key, None)
                if items is not None:
                    saved_states = ckpt[key + "_state_dict"]
                    for obj, state in zip(items, saved_states):
                        if obj is not None:
                            obj.load_state_dict(state)

        return trainer


