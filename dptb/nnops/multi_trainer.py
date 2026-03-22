import torch
import torch.nn as nn
import logging
from typing import Union

from dptb.utils.tools import get_lr_scheduler, get_optimizer
from dptb.data import AtomicDataset, AtomicData
from dptb.data.AtomicDataDict import with_edge_vectors
from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model

log = logging.getLogger(__name__)


class MultiTrainer(Trainer):
    """
    基于距离划分的 MOE Trainer (多专家训练器)。

    支持两种执行模式：
    1) serial：保守模式，逐 expert 串行 forward/backward/step
    2) parallel_multi：全链路异步并行模式 (Full Pipeline Async)
       - CUDA: 使用多 stream 同时发起多个 expert 的 Forward + Backward + Step，极致榨干 GPU。
       - 非 CUDA: 退化为顺序全流程执行。
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

        log.info(
            f"🚀 Initialized MultiTrainer with {self.num_experts} ISOLATED experts (Distance MOE): {self.distance_ranges}"
        )

        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

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

        if self.parallel_multi:
            if self._is_cuda_device() and self.num_experts > 1:
                log.info("⚡ Expert mode = parallel_multi (Full Async: Fwd+Bwd+Step via CUDA streams)")
            elif self.num_experts > 1:
                log.info("⚡ Expert mode = parallel_multi (Sequential fallback, non-CUDA)")
            else:
                log.info("⚡ Expert mode = serial (only one expert)")
        else:
            log.info("⚡ Expert mode = serial")

    def _device_obj(self):
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda_device(self):
        return self._device_obj().type == "cuda"

    def _use_cuda_stream_parallel(self):
        return self.parallel_multi and self.num_experts > 1 and self._is_cuda_device()

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
        batch_dev = batch.to(self.device)
        batch_info = {
            "__slices__": batch_dev.__slices__,
            "__cumsum__": batch_dev.__cumsum__,
            "__cat_dims__": batch_dev.__cat_dims__,
            "__num_nodes_list__": batch_dev.__num_nodes_list__,
            "__data_class__": batch_dev.__data_class__,
        }
        batch_dict = AtomicData.to_AtomicDataDict(batch_dev)
        if with_lengths:
            batch_dict = with_edge_vectors(batch_dict, with_lengths=True)
        return batch_dict, batch_info

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
                if isinstance(inner, nn.Module) or any(
                        hasattr(inner, key) for key in (
                                "last_onsite_loss", "last_hopping_loss", "last_z_loss", "expert_load_cv"
                        )
                ):
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
            return float(value.detach().item())
        return float(value)

    def _to_int_scalar(self, value, default=0):
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            return int(value.detach().item())
        return int(value)

    def _snapshot_loss_metrics(self, loss_obj):
        loss_module = self._resolve_loss_module(loss_obj)
        return {
            "onsite": self._as_scalar_tensor(getattr(loss_module, "last_onsite_loss", 0.0), default=0.0),
            "hopping": self._as_scalar_tensor(getattr(loss_module, "last_hopping_loss", 0.0), default=0.0),
            "z_loss": self._as_scalar_tensor(getattr(loss_module, "last_z_loss", None), allow_none=True),
            "expert_load_cv": self._as_scalar_tensor(getattr(loss_module, "expert_load_cv", None), allow_none=True),
        }

    def _run_one_expert_loss(self, batch_dict, batch_info, criterion, expert_idx, range_dis, capture_metrics=False):
        expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis, expert_idx)

        batch_copy = batch_dict.copy()
        batch_copy["expert_edge_mask"] = expert_edge_mask
        batch_copy["expert_node_mask"] = expert_node_mask
        batch_copy["expert_idx"] = torch.tensor(expert_idx, dtype=torch.long, device=self.device)

        batch_for_loss = batch_copy.copy()
        pred_batch = self.model(batch_copy)

        pred_batch.update(batch_info)
        batch_for_loss.update(batch_info)

        loss = criterion(pred_batch, batch_for_loss)

        out = {
            "loss": loss,
            "active_nodes": expert_node_mask.sum().detach(),
            "active_edges": expert_edge_mask.sum().detach(),
        }

        if capture_metrics:
            out.update(self._snapshot_loss_metrics(criterion))

        return out

    def _build_train_payload(self, batch_dict, batch_info, expert_idx, range_dis,
                             ref_batch_dict=None, ref_batch_info=None):
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
            hopping_weighted_sum = hopping_weighted_sum + ref_res["hopping"] * ref_res["active_edges"].to(
                dtype=self.dtype)

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
            "z_values": z_values,
            "load_cv_values": load_cv_values,
        }

    def _launch_train_payloads_parallel(self, batch_dict, batch_info, ref_batch_dict=None, ref_batch_info=None):
        payloads = [None] * self.num_experts

        if self._use_cuda_stream_parallel():
            device = self._device_obj()
            base_stream = torch.cuda.current_stream(device=device)
            streams = [torch.cuda.Stream(device=device) for _ in range(self.num_experts)]

            # 1. 确保所有新流等待主数据准备完成
            for s in streams:
                s.wait_stream(base_stream)

            # 2. 纯粹并行的异步下发 Fwd -> Bwd -> Step -> Clean
            for expert_idx, range_dis in enumerate(self.distance_ranges):
                with torch.cuda.stream(streams[expert_idx]):
                    # 【重要修正 3】起步确保无残留梯度
                    self.optimizers[expert_idx].zero_grad(set_to_none=True)

                    payload = self._build_train_payload(
                        batch_dict=batch_dict,
                        batch_info=batch_info,
                        expert_idx=expert_idx,
                        range_dis=range_dis,
                        ref_batch_dict=ref_batch_dict,
                        ref_batch_info=ref_batch_info,
                    )

                    loss_expert = payload["loss"]
                    loss_expert.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.experts[expert_idx].parameters(),
                        max_norm=self.clip_grad_norm
                    )
                    self.optimizers[expert_idx].step()

                    # 【重要修正 3】Step 完立刻释放梯度显存（极其节约 VRAM）
                    self.optimizers[expert_idx].zero_grad(set_to_none=True)

                    # 【重要修正 1】移除了此处的 _scheduler_step，防止内部 .item() 强行阻塞并发流

                    # 【重要修正 2】切断计算图根节点引用，保持干净的返回包
                    payload["loss"] = loss_expert.detach()

                    payload["grad_norm"] = grad_norm
                    payloads[expert_idx] = payload

            # 3. 阻塞主线程，等待所有 GPU 计算流完结 (此时可以安全收集 metrics)
            current = torch.cuda.current_stream(device=device)
            for s in streams:
                current.wait_stream(s)

        else:
            # 串行 Fallback
            for expert_idx, range_dis in enumerate(self.distance_ranges):
                self.optimizers[expert_idx].zero_grad(set_to_none=True)

                payload = self._build_train_payload(
                    batch_dict=batch_dict,
                    batch_info=batch_info,
                    expert_idx=expert_idx,
                    range_dis=range_dis,
                    ref_batch_dict=ref_batch_dict,
                    ref_batch_info=ref_batch_info,
                )

                loss_expert = payload["loss"]
                loss_expert.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.experts[expert_idx].parameters(),
                    max_norm=self.clip_grad_norm
                )
                self.optimizers[expert_idx].step()
                self.optimizers[expert_idx].zero_grad(set_to_none=True)

                payload["loss"] = loss_expert.detach()
                payload["grad_norm"] = grad_norm
                payloads[expert_idx] = payload

        return payloads

    def _launch_validation_losses_parallel(self, batch_dict, batch_info):
        losses = [None] * self.num_experts

        if self._use_cuda_stream_parallel():
            device = self._device_obj()
            base_stream = torch.cuda.current_stream(device=device)
            streams = [torch.cuda.Stream(device=device) for _ in range(self.num_experts)]

            for s in streams:
                s.wait_stream(base_stream)

            for expert_idx, range_dis in enumerate(self.distance_ranges):
                with torch.cuda.stream(streams[expert_idx]):
                    losses[expert_idx] = self._run_one_expert_loss(
                        batch_dict=batch_dict,
                        batch_info=batch_info,
                        criterion=self.validation_lossfunc,
                        expert_idx=expert_idx,
                        range_dis=range_dis,
                        capture_metrics=False
                    )["loss"].detach()

            current = torch.cuda.current_stream(device=device)
            for s in streams:
                current.wait_stream(s)
        else:
            for expert_idx, range_dis in enumerate(self.distance_ranges):
                losses[expert_idx] = self._run_one_expert_loss(
                    batch_dict=batch_dict,
                    batch_info=batch_info,
                    criterion=self.validation_lossfunc,
                    expert_idx=expert_idx,
                    range_dis=range_dis,
                    capture_metrics=False
                )["loss"].detach()

        return losses

    def _scheduler_step(self, expert_idx, loss_expert):
        """
        注意：传入的 loss_expert 必须已经是 .detach() 的，防止意外的内存泄漏或图反向。
        这里的 .item() 在 wait_stream() 后执行是安全的，不会阻塞并发调度。
        """
        if self.update_lr_per_iter:
            if isinstance(self.lr_schedulers[expert_idx], torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.iter > 1:
                    # 确保是 Python float
                    val = loss_expert.item() if torch.is_tensor(loss_expert) else float(loss_expert)
                    self.lr_schedulers[expert_idx].step(val)
            else:
                self.lr_schedulers[expert_idx].step()

    def iteration(self, batch, ref_batch=None):
        self.model.train()

        batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

        ref_batch_dict = None
        ref_batch_info = None
        if ref_batch is not None:
            ref_batch_dict, ref_batch_info = self._prepare_batch_bundle(ref_batch, with_lengths=True)

        total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
        expert_grad_norms = []

        global_onsite_sum = 0.0
        global_hopping_sum = 0.0
        total_active_nodes = 0
        total_active_edges = 0
        expert_onsite_dict = {}
        expert_hopping_dict = {}
        z_metric_values = []
        expert_load_cv_values = []

        def collect_payload(expert_idx, payload, grad_norm):
            nonlocal total_loss
            nonlocal global_onsite_sum, global_hopping_sum
            nonlocal total_active_nodes, total_active_edges

            total_loss = total_loss + payload["loss"]

            expert_grad_norms.append(self._to_float_scalar(grad_norm))

            expert_onsite = self._to_float_scalar(payload["expert_onsite"])
            expert_hopping = self._to_float_scalar(payload["expert_hopping"])
            expert_onsite_dict[f"expert_{expert_idx}"] = expert_onsite
            expert_hopping_dict[f"expert_{expert_idx}"] = expert_hopping

            global_onsite_sum += self._to_float_scalar(payload["onsite_weighted_sum"])
            global_hopping_sum += self._to_float_scalar(payload["hopping_weighted_sum"])
            total_active_nodes += self._to_int_scalar(payload["active_nodes"])
            total_active_edges += self._to_int_scalar(payload["active_edges"])

            for z in payload["z_values"]:
                if z is not None:
                    z_metric_values.append(self._to_float_scalar(z))

            for cv in payload["load_cv_values"]:
                if cv is not None:
                    expert_load_cv_values.append(self._to_float_scalar(cv))

        # 核心：无论是不是并行，逻辑收束到统一的数据处理流
        payloads = self._launch_train_payloads_parallel(
            batch_dict=batch_dict,
            batch_info=batch_info,
            ref_batch_dict=ref_batch_dict,
            ref_batch_info=ref_batch_info,
        )

        for expert_idx, payload in enumerate(payloads):
            # 1. 收集统计信息
            collect_payload(expert_idx, payload, payload["grad_norm"])
            # 2. 【重要修正 1】安全、统一地在主进程收集完数据后执行 Scheduler Step
            self._scheduler_step(expert_idx, payload["loss"])

        global_onsite = global_onsite_sum / max(total_active_nodes, 1)
        global_hopping = global_hopping_sum / max(total_active_edges, 1)

        state = {
            'field': 'iteration',
            "train_loss": total_loss,
            "lr": self.optimizers[0].param_groups[0]['lr'],
            "total_grad_norm": sum(expert_grad_norms) / max(len(expert_grad_norms), 1),
            "train_onsite_loss": global_onsite,
            "train_hopping_loss": global_hopping,
        }

        for i in range(self.num_experts):
            state[f"expert_{i}_onsite"] = expert_onsite_dict.get(f"expert_{i}", 0.0)
            state[f"expert_{i}_hopping"] = expert_hopping_dict.get(f"expert_{i}", 0.0)

        if expert_load_cv_values:
            state["expert_load_cv"] = sum(expert_load_cv_values) / len(expert_load_cv_values)
        if z_metric_values:
            state["mean_max_prob"] = sum(z_metric_values) / len(z_metric_values)

        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return total_loss

    def validation(self, fast=True):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            self.model.eval()

            for batch in self.validation_loader:
                batch_dict, batch_info = self._prepare_batch_bundle(batch, with_lengths=True)

                losses = self._launch_validation_losses_parallel(
                    batch_dict=batch_dict,
                    batch_info=batch_info
                )

                for loss_i in losses:
                    total_loss = total_loss + loss_i

                if fast:
                    break

        if not fast:
            total_loss = total_loss / len(self.validation_loader)

        return total_loss

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