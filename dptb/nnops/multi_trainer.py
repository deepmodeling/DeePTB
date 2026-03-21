import torch
import torch.nn as nn
import logging
from typing import Union

from dptb.utils.tools import get_lr_scheduler, get_optimizer
from dptb.data import AtomicDataset, AtomicData
from dptb.nnops.trainer import Trainer
from dptb.nn.build import build_model

log = logging.getLogger(__name__)


class MultiTrainer(Trainer):
    """
    基于距离划分的 MOE Trainer (多专家训练器)。
    实现物理隔离：每个 expert 拥有独立的 optimizer、lr_scheduler，独立计算 forward、backward 和 clip_grad。
    """
    # 覆盖父类的 object_keys，使得 Saver 插件能够识别并保存列表类型的状态
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

        # 1. 调用父类初始化
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
        log.info(
            f"🚀 Initialized MultiTrainer with {self.num_experts} ISOLATED experts (Distance MOE): {self.distance_ranges}")

        # 2. 校验模型是否具备 ModuleList 类型的 experts
        if not hasattr(self.model, 'experts') or len(self.model.experts) != self.num_experts:
            raise ValueError(f"Model must have a nn.ModuleList named 'experts' with {self.num_experts} sub-models!")

        # 3. 构建完全独立的 Optimizers 和 Schedulers
        self.optimizers = []
        self.lr_schedulers = []
        for i in range(self.num_experts):
            opt = get_optimizer(model_param=self.model.experts[i].parameters(), **self.train_options["optimizer"])
            sch = get_lr_scheduler(optimizer=opt, **self.train_options["lr_scheduler"])
            self.optimizers.append(opt)
            self.lr_schedulers.append(sch)

        # 4. 删除父类遗留的单一优化器
        if hasattr(self, "optimizer"): del self.optimizer
        if hasattr(self, "lr_scheduler"): del self.lr_scheduler

    def _prepare_expert_masks(self, batch_dict, range_dis):
        """
        Helper 函数：根据当前专家的距离区间，预生成物理距离掩码 (Mask)。
        """
        d_min, d_max = range_dis

        # 1. 边掩码 (Edge Mask)
        # 根据 edge_vec 的 L2 范数（即距离）生成布尔掩码
        edge_vec = batch_dict["edge_vec"]
        dist = torch.norm(edge_vec, dim=-1)
        expert_edge_mask = (dist >= d_min) & (dist < d_max)

        # 2. 节点掩码 (Node Mask)
        # 物理规则：只有包含 d=0 的专家才负责 Onsite (节点) 损失
        num_nodes = batch_dict["node_features"].shape[0]
        expert_node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)

        # 如果起始距离大于 0，说明该专家不负责 Onsite，将所有节点屏蔽
        if d_min > 0:
            expert_node_mask.fill_(False)

        return expert_edge_mask, expert_node_mask

    def iteration(self, batch, ref_batch=None):
        self.model.train()
        batch_dev = batch.to(self.device)

        batch_info = {
            "__slices__": batch_dev.__slices__, "__cumsum__": batch_dev.__cumsum__,
            "__cat_dims__": batch_dev.__cat_dims__, "__num_nodes_list__": batch_dev.__num_nodes_list__,
            "__data_class__": batch_dev.__data_class__,
        }
        batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

        total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
        expert_losses = []
        expert_grad_norms = []

        for expert_idx, range_dis in enumerate(self.distance_ranges):
            self.optimizers[expert_idx].zero_grad(set_to_none=True)

            # 生成并注入当前专家的物理掩码
            expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis)

            batch_copy = batch_dict.copy()
            batch_copy["expert_edge_mask"] = expert_edge_mask
            batch_copy["expert_node_mask"] = expert_node_mask
            batch_copy["expert_idx"] = expert_idx

            batch_for_loss = batch_copy.copy()
            pred_batch = self.model(batch_copy)  # 路由到对应专家

            pred_batch.update(batch_info)
            batch_for_loss.update(batch_info)

            # 计算 Loss，此时 Loss 内部会使用注入的 Mask
            loss_expert = self.train_lossfunc(pred_batch, batch_for_loss)

            # 处理 Reference Batch
            if ref_batch is not None:
                ref_batch_dev = ref_batch.to(self.device)
                ref_batch_info = {
                    "__slices__": ref_batch_dev.__slices__, "__cumsum__": ref_batch_dev.__cumsum__,
                    "__cat_dims__": ref_batch_dev.__cat_dims__, "__num_nodes_list__": ref_batch_dev.__num_nodes_list__,
                    "__data_class__": ref_batch_dev.__data_class__,
                }
                ref_batch_dict = AtomicData.to_AtomicDataDict(ref_batch_dev)

                # 为 Reference Batch 也生成掩码
                ref_e_mask, ref_n_mask = self._prepare_expert_masks(ref_batch_dict, range_dis)

                ref_batch_copy = ref_batch_dict.copy()
                ref_batch_copy["expert_edge_mask"] = ref_e_mask
                ref_batch_copy["expert_node_mask"] = ref_n_mask
                ref_batch_copy["expert_idx"] = expert_idx

                ref_batch_for_loss = ref_batch_copy.copy()
                pred_ref = self.model(ref_batch_copy)
                pred_ref.update(ref_batch_info)
                ref_batch_for_loss.update(ref_batch_info)

                loss_expert += self.train_lossfunc(pred_ref, ref_batch_for_loss)

            loss_expert.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.experts[expert_idx].parameters(),
                max_norm=self.clip_grad_norm
            )
            self.optimizers[expert_idx].step()

            if self.update_lr_per_iter:
                if isinstance(self.lr_schedulers[expert_idx], torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.iter > 1:
                        self.lr_schedulers[expert_idx].step(loss_expert.detach())
                else:
                    self.lr_schedulers[expert_idx].step()

            total_loss += loss_expert.detach()
            expert_losses.append(loss_expert.detach())
            expert_grad_norms.append(grad_norm.item())

        loss_obj = self.train_lossfunc
        for attr in ("lossfunc", "loss_fn", "criterion", "method", "loss"):
            inner = getattr(loss_obj, attr, None)
            if isinstance(inner, nn.Module):
                loss_obj = inner
                break

        state = {
            'field': 'iteration',
            "train_loss": total_loss,
            "expert_losses": expert_losses,
            "lr": self.optimizers[0].state_dict()["param_groups"][0]['lr'],
            "total_grad_norm": sum(expert_grad_norms) / self.num_experts
        }

        onsite_comp = getattr(loss_obj, "last_onsite_loss", None)
        hopping_comp = getattr(loss_obj, "last_hopping_loss", None)
        z_loss_comp = getattr(loss_obj, "last_z_loss", None)
        expert_load_cv = getattr(loss_obj, "expert_load_cv", None)

        if onsite_comp is not None: state["train_onsite_loss"] = onsite_comp
        if hopping_comp is not None: state["train_hopping_loss"] = hopping_comp
        if expert_load_cv is not None: state["expert_load_cv"] = expert_load_cv
        if z_loss_comp is not None: state["mean_max_prob"] = z_loss_comp

        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return total_loss

    def validation(self, fast=True):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            self.model.eval()
            for batch in self.validation_loader:
                batch_dev = batch.to(self.device)
                batch_info = {"__slices__": batch_dev.__slices__, "__cumsum__": batch_dev.__cumsum__,
                              "__cat_dims__": batch_dev.__cat_dims__,
                              "__num_nodes_list__": batch_dev.__num_nodes_list__,
                              "__data_class__": batch_dev.__data_class__}
                batch_dict = AtomicData.to_AtomicDataDict(batch_dev)

                for expert_idx, range_dis in enumerate(self.distance_ranges):
                    expert_edge_mask, expert_node_mask = self._prepare_expert_masks(batch_dict, range_dis)

                    batch_copy = batch_dict.copy()
                    batch_copy["expert_edge_mask"] = expert_edge_mask
                    batch_copy["expert_node_mask"] = expert_node_mask
                    batch_copy["expert_idx"] = expert_idx

                    batch_for_loss = batch_copy.copy()
                    pred_batch = self.model(batch_copy)
                    pred_batch.update(batch_info)
                    batch_for_loss.update(batch_info)

                    total_loss += self.validation_lossfunc(pred_batch, batch_for_loss)

                if fast: break

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
        if len(train_options) == 0: train_options = ckpt["config"]["train_options"]
        if len(common_options) == 0: common_options = ckpt["config"]["common_options"]

        distance_ranges = train_options.get("distance_ranges", [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]])

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