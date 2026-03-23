import contextlib
import time
import logging
from typing import Union, Optional, Dict, Any, List

import torch
import torch.nn as nn

from dptb.utils.tools import get_lr_scheduler, get_optimizer
from dptb.data import AtomicDataset, AtomicData
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model

log = logging.getLogger(__name__)


# =============================================================================
# TAGGER
# =============================================================================

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
        mem0 = self._cuda_mem() if self.cuda_mem else None

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


# =============================================================================
# MultiTrainer
# =============================================================================

class MultiTrainer(Trainer):
    """
    距离 MOE 多专家 Trainer。

    重点实现：共享 scheduler（尤其 ReduceLROnPlateau）
    - 并行多 stream：每个 expert stream 做 Fwd/Loss/Bwd/Clip/Step
    - batch 结束 barrier(wait_streams) 后，再 .item() + scheduler.step(metric_float)
    - 所有 expert 的 scheduler 使用同一个 metric

    可比口径 train_loss：
    - 默认 mode="reduce"：通过 loss 暴露的 sum/count 严格还原 stitched loss（不做额外 stitched forward）
    """

    object_keys = ["lr_schedulers", "optimizers"]

    def __init__(
        self,
        distance_ranges: list,
        train_options: dict,
        common_options: dict,
        model: torch.nn.Module,
        train_datasets: AtomicDataset,
        reference_datasets: Union[AtomicDataset, None] = None,
        validation_datasets: Union[AtomicDataset, None] = None,
    ) -> None:

        super().__init__(
            train_options=train_options,
            common_options=common_options,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
        )

        self.distance_ranges = distance_ranges
        self.num_experts = len(distance_ranges)
        self.parallel_multi = bool(self.train_options.get("parallel_multi", False))

        # debug/tag options
        self.debug_tags = bool(self.train_options.get("debug_tags", False))
        self.debug_tag_freq = int(self.train_options.get("debug_tag_freq", 1))
        self.debug_tag_cuda_mem = bool(self.train_options.get("debug_tag_cuda_mem", True))
        self.debug_tag_cuda_sync = bool(self.train_options.get("debug_tag_cuda_sync", False))
        self.debug_oom_dump = bool(self.train_options.get("debug_oom_dump", True))

        self._tagger = _StageTagger(
            trainer=self,
            enabled=self.debug_tags,
            freq=self.debug_tag_freq,
            cuda_mem=self.debug_tag_cuda_mem,
            cuda_sync=self.debug_tag_cuda_sync,
            oom_dump=self.debug_oom_dump,
        )

        # comparable loss options
        self.log_single_model_compatible_loss = bool(
            self.train_options.get("log_single_model_compatible_loss", True)
        )
        self.log_single_model_compatible_loss_mode = str(
            self.train_options.get("log_single_model_compatible_loss_mode", "reduce")
        ).lower()  # "reduce" | "full_forward" | "off"

        # shared scheduler options
        # metric source: "train_loss_opt" (default, true objective) or "train_loss" (comparable)
        self.shared_scheduler_metric = str(
            self.train_options.get("shared_scheduler_metric", "train_loss_opt")
        ).lower()

        log.info(
            f"[MultiTrainer] num_experts={self.num_experts}, parallel_multi={self.parallel_multi}, "
            f"log_single_model_compatible_loss={self.log_single_model_compatible_loss}, "
            f"mode={self.log_single_model_compatible_loss_mode}, "
            f"shared_scheduler_metric={self.shared_scheduler_metric}"
        )

        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

        # optimizers/schedulers per expert
        self.optimizers = []
        self.lr_schedulers = []
        for i in range(self.num_experts):
            opt = get_optimizer(model_param=self.model.experts[i].parameters(), **self.train_options["optimizer"])
            sch = get_lr_scheduler(optimizer=opt, **self.train_options["lr_scheduler"])
            self.optimizers.append(opt)
            self.lr_schedulers.append(sch)

        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "lr_scheduler"):
            del self.lr_scheduler

        self._warn_non_expert_trainables()
        self._t_last_iter_end: Optional[float] = None

    # -------------------------------------------------------------------------
    # device helpers
    # -------------------------------------------------------------------------

    def _device_obj(self):
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda_device(self):
        return self._device_obj().type == "cuda" and torch.cuda.is_available()

    def _use_cuda_stream_parallel(self):
        return self.parallel_multi and self.num_experts > 1 and self._is_cuda_device()

    # -------------------------------------------------------------------------
    # sanity checks
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # masks & batch prep
    # -------------------------------------------------------------------------

    def _prepare_expert_masks(self, batch_dict, range_dis, expert_idx):
        d_min, d_max = range_dis
        dist = batch_dict['edge_lengths']

        if expert_idx == self.num_experts - 1:
            expert_edge_mask = (dist >= d_min)
        else:
            expert_edge_mask = (dist >= d_min) & (dist < d_max)

        num_nodes = batch_dict["node_features"].shape[0]
        expert_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
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

    # -------------------------------------------------------------------------
    # loss metric extraction helpers
    # -------------------------------------------------------------------------

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
            return out

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
        """
        legacy + strict reduce stats (if loss supports)
        """
        loss_module = self._resolve_loss_module(loss_obj)

        out = {
            # legacy monitors
            "onsite": self._as_scalar_tensor(getattr(loss_module, "last_onsite_loss", 0.0), default=0.0),
            "hopping": self._as_scalar_tensor(getattr(loss_module, "last_hopping_loss", 0.0), default=0.0),
            "z_loss": self._as_scalar_tensor(getattr(loss_module, "last_z_loss", None), allow_none=True),
            "expert_load_cv": self._as_scalar_tensor(getattr(loss_module, "expert_load_cv", None), allow_none=True),
        }

        # strict reduce stats (optional)
        for k in (
            "last_onsite_l1_sum", "last_onsite_mse_sum", "last_onsite_count",
            "last_hopping_l1_sum", "last_hopping_mse_sum", "last_hopping_count",
        ):
            v = getattr(loss_module, k, None)
            out[k] = self._as_scalar_tensor(v, default=0.0) if v is not None else None

        return out

    # -------------------------------------------------------------------------
    # core fwd/loss per expert
    # -------------------------------------------------------------------------

    def _run_one_expert_loss(self, batch_dict, batch_info, criterion, expert_idx, range_dis, capture_metrics=False):
        with self._tagger.tag("expert/prepare_masks", it=self.iter, expert=expert_idx):
            expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)

        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask

        # IMPORTANT: use python int (avoid CUDA tensor -> int bug in wrapper)
        batch_copy["expert_idx"] = int(expert_idx)

        with self._tagger.tag("expert/model_forward", it=self.iter, expert=expert_idx):
            pred_batch = self.model(batch_copy)

        # pass global step so loss doesn't advance per-expert (if loss supports global_step)
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
        ref_batch_dict=None, ref_batch_info=None
    ):
        main = self._run_one_expert_loss(
            batch_dict=batch_dict,
            batch_info=batch_info,
            criterion=self.train_lossfunc,
            expert_idx=expert_idx,
            range_dis=range_dis,
            capture_metrics=True
        )

        total_loss = main["loss"]
        active_nodes = main["active_nodes"]
        active_edges = main["active_edges"]

        onsite_weighted_sum = main["onsite"] * active_nodes.to(dtype=self.dtype)
        hopping_weighted_sum = main["hopping"] * active_edges.to(dtype=self.dtype)

        # strict reduce stats (optional)
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
                criterion=self.train_lossfunc,
                expert_idx=expert_idx,
                range_dis=range_dis,
                capture_metrics=True
            )

            total_loss = total_loss + ref_res["loss"]
            active_nodes = active_nodes + ref_res["active_nodes"]
            active_edges = active_edges + ref_res["active_edges"]
            onsite_weighted_sum = onsite_weighted_sum + ref_res["onsite"] * ref_res["active_nodes"].to(dtype=self.dtype)
            hopping_weighted_sum = hopping_weighted_sum + ref_res["hopping"] * ref_res["active_edges"].to(dtype=self.dtype)

            # strict reduce stats add (if enabled)
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

            # legacy monitors
            "expert_onsite": expert_onsite.detach(),
            "expert_hopping": expert_hopping.detach(),
            "onsite_weighted_sum": onsite_weighted_sum.detach(),
            "hopping_weighted_sum": hopping_weighted_sum.detach(),
            "active_nodes": active_nodes.detach(),
            "active_edges": active_edges.detach(),

            # strict reduce stats (detached)
            "onsite_l1_sum": onsite_l1_sum.detach() if torch.is_tensor(onsite_l1_sum) else None,
            "onsite_mse_sum": onsite_mse_sum.detach() if torch.is_tensor(onsite_mse_sum) else None,
            "onsite_cnt": onsite_cnt.detach() if torch.is_tensor(onsite_cnt) else None,
            "hopping_l1_sum": hopping_l1_sum.detach() if torch.is_tensor(hopping_l1_sum) else None,
            "hopping_mse_sum": hopping_mse_sum.detach() if torch.is_tensor(hopping_mse_sum) else None,
            "hopping_cnt": hopping_cnt.detach() if torch.is_tensor(hopping_cnt) else None,

            "z_values": [z.detach() for z in z_values],
            "load_cv_values": [cv.detach() for cv in load_cv_values],
        }

    # -------------------------------------------------------------------------
    # stitched-compatible loss reconstruction (reduce)
    # -------------------------------------------------------------------------

    def _compute_stitched_loss_by_reduce(self, payloads: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
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

        eps = torch.tensor(1.0, dtype=self.dtype, device=self.device)

        def _safe_mean(sum_t, cnt_t):
            if sum_t is None or cnt_t is None:
                return torch.zeros((), dtype=self.dtype, device=self.device)
            return sum_t / torch.clamp(cnt_t.to(dtype=self.dtype), min=eps)

        onsite_l1_mean = _safe_mean(onsite_l1_sum, onsite_cnt)
        onsite_mse_mean = _safe_mean(onsite_mse_sum, onsite_cnt)
        hopping_l1_mean = _safe_mean(hopping_l1_sum, hopping_cnt)
        hopping_mse_mean = _safe_mean(hopping_mse_sum, hopping_cnt)

        onsite_loss = 0.5 * (onsite_l1_mean + torch.sqrt(onsite_mse_mean))
        hopping_loss = 0.5 * (hopping_l1_mean + torch.sqrt(hopping_mse_mean))

        # onsite_boost and z_loss coef from the actual loss module
        loss_module = self._resolve_loss_module(self.train_lossfunc)
        onsite_boost = bool(getattr(loss_module, "onsite_boost", False))
        onsite_boost_w = float(getattr(loss_module, "_current_onsite_weight", lambda: 1.0)())
        z_coef = float(getattr(loss_module, "z_loss_coef", 0.0))

        if onsite_boost:
            total = onsite_boost_w * onsite_loss + hopping_loss
        else:
            total = 0.5 * (onsite_loss + hopping_loss)

        # z_loss: reduce 无法严格等价 stitched 的 mean_max_prob，只能取一个代表值
        if z_coef > 0.0 and len(z_vals) > 0:
            total = total + z_coef * z_vals[0]

        return total.detach()

    # -------------------------------------------------------------------------
    # parallel launch (NO scheduler inside streams)
    # -------------------------------------------------------------------------

    def _launch_train_payloads_parallel(self, batch_dict, batch_info, ref_batch_dict=None, ref_batch_info=None):
        payloads = [None] * self.num_experts

        if self._use_cuda_stream_parallel():
            device = self._device_obj()
            base_stream = torch.cuda.current_stream(device=device)
            streams = [torch.cuda.Stream(device=device) for _ in range(self.num_experts)]

            with self._tagger.tag("parallel/streams_wait_base", it=self.iter):
                for s in streams:
                    s.wait_stream(base_stream)

            for expert_idx, range_dis in enumerate(self.distance_ranges):
                with torch.cuda.stream(streams[expert_idx]):
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

                    payloads[expert_idx] = payload

            # barrier: after this, accessing loss_detached.item() should not add extra waiting
            with self._tagger.tag("parallel/wait_streams(barrier)", it=self.iter):
                current = torch.cuda.current_stream(device=device)
                for s in streams:
                    current.wait_stream(s)

        else:
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

                payloads[expert_idx] = payload

        return payloads

    # -------------------------------------------------------------------------
    # shared scheduler step (after barrier)
    # -------------------------------------------------------------------------

    def _shared_scheduler_step_after_barrier(self, metric_tensor: torch.Tensor):
        """
        正确做法：
        - barrier 之后才 .item()
        - 所有 scheduler 使用同一个 metric_float
        """
        if not self.update_lr_per_iter:
            return

        with self._tagger.tag("scheduler/metric_item", it=self.iter, extra=f"type={self.shared_scheduler_metric}"):
            if torch.is_tensor(metric_tensor):
                m = metric_tensor.detach()
                if m.ndim != 0:
                    m = m.mean()
                metric_float = float(m.item())
            else:
                metric_float = float(metric_tensor)

        with self._tagger.tag("scheduler/shared_step(barrier_after)", it=self.iter, extra=f"metric={metric_float:.6g}"):
            for expert_idx, sch in enumerate(self.lr_schedulers):
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.iter > 1:
                        sch.step(metric_float)
                else:
                    sch.step()

    # -------------------------------------------------------------------------
    # iteration
    # -------------------------------------------------------------------------

    def iteration(self, batch, ref_batch=None):
        # rough "outside-iteration" wait time
        t_now = time.perf_counter()
        if self._t_last_iter_end is not None and self.debug_tags and (self.iter % self.debug_tag_freq == 0):
            log.info(f"[TAG][it={self.iter}][data_wait(outside_iteration)] dt={(t_now - self._t_last_iter_end):.4f}s")

        with self._tagger.tag("iteration/entry", it=self.iter):
            self.model.train()

        try:
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

            with self._tagger.tag("iteration/train_experts(parallel_or_serial)", it=self.iter,
                                  extra=f"parallel_multi={self.parallel_multi}"):
                payload_list = self._launch_train_payloads_parallel(
                    batch_dict=batch_dict,
                    batch_info=batch_info,
                    ref_batch_dict=ref_batch_dict,
                    ref_batch_info=ref_batch_info,
                )

            with self._tagger.tag("iteration/collect_payloads", it=self.iter):
                for expert_idx, payload in enumerate(payload_list):
                    collect_payload(expert_idx, payload)

            global_onsite = global_onsite_sum / max(total_active_nodes, 1)
            global_hopping = global_hopping_sum / max(total_active_edges, 1)

            with self._tagger.tag("iteration/compute_train_loss_compatible(reduce)", it=self.iter):
                comparable_train_loss = self._compute_stitched_loss_by_reduce(reduce_payloads)

            # choose final train_loss for logging
            final_train_loss = comparable_train_loss if comparable_train_loss is not None else total_loss_opt

            # choose shared scheduler metric tensor
            if self.shared_scheduler_metric == "train_loss":
                sched_metric = final_train_loss
            else:
                # default: true optimized objective
                sched_metric = total_loss_opt

            # shared scheduler step AFTER barrier
            self._shared_scheduler_step_after_barrier(sched_metric)

            state = {
                'field': 'iteration',
                "train_loss": final_train_loss,          # comparable if available
                "train_loss_opt": total_loss_opt,        # true objective (sum experts)
                "lr": self.optimizers[0].param_groups[0]['lr'],
                "total_grad_norm": sum(expert_grad_norms) / max(len(expert_grad_norms), 1),
                "train_onsite_loss": global_onsite,
                "train_hopping_loss": global_hopping,
            }

            for i in range(self.num_experts):
                state[f"expert_{i}_onsite"] = expert_onsite_dict.get(f"expert_{i}_onsite", 0.0)
                state[f"expert_{i}_hopping"] = expert_hopping_dict.get(f"expert_{i}_hopping", 0.0)

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

    # -------------------------------------------------------------------------
    # validation (kept simple; uses reduce if enabled)
    # -------------------------------------------------------------------------

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

                # reduce path (preferred)
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

                    # temporarily use validation loss module for boost/z_coef logic
                    old = self.train_lossfunc
                    self.train_lossfunc = self.validation_lossfunc
                    with self._tagger.tag("validation/compute_reduce_loss", it=self.iter):
                        loss_i = self._compute_stitched_loss_by_reduce(payloads)
                    self.train_lossfunc = old

                    if loss_i is None:
                        # fallback
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

    # -------------------------------------------------------------------------
    # restart
    # -------------------------------------------------------------------------

    @classmethod
    def restart(cls, checkpoint, train_datasets, train_options={}, common_options={}, reference_datasets=None,
                validation_datasets=None):
        ckpt = torch.load(checkpoint, map_location=common_options["device"], weights_only=False)
        model = build_model(
            checkpoint=checkpoint,
            model_options=ckpt["config"]["model_options"],
            common_options=ckpt["config"]["common_options"],
            train_options=ckpt["config"].get("train_options", train_options)
        )
        if len(train_options) == 0:
            train_options = ckpt["config"]["train_options"]
        if len(common_options) == 0:
            common_options = ckpt["config"]["common_options"]

        distance_ranges = train_options.get(
            "distance_ranges",
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
        )

        trainer = cls(
            distance_ranges=distance_ranges,
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
            train_options=train_options,
            common_options=common_options
        )

        trainer.ep = ckpt["epoch"] + 1
        trainer.iter = ckpt["iteration"] + 1
        trainer.stats = ckpt["stats"]

        queues_name = list(trainer.plugin_queues.keys())
        for unit in queues_name:
            for plugin in trainer.plugin_queues[unit]:
                plugin = (getattr(trainer, unit) + plugin[0], plugin[1], plugin[2])

        for key in cls.object_keys:
            items = getattr(trainer, key, None)
            if items is not None:
                saved_states = ckpt[key + "_state_dict"]
                for obj, state in zip(items, saved_states):
                    obj.load_state_dict(state)

        return trainer