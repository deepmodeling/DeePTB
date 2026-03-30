import shutil
from dptb.plugins.base_plugin import Plugin
import logging
import os
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


class Saver(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Saver, self).__init__(interval)
        self.best_loss = 1e7
        self.best_quene = []
        self.latest_quene = []

    def register(self, trainer, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.trainer = trainer

        if self.trainer.model.name == "nnsk":
            push_option = self.trainer.model.model_options["nnsk"].get("push", False)
            if push_option:
                if abs(push_option['rs_thr']) + abs(push_option['w_thr']) != 0.0 and abs(push_option['ovp_thr']) != 0.0:
                    log.error("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")
                    raise ValueError("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")

                if abs(push_option['rs_thr']) + abs(push_option['w_thr']) != 0.0:
                    push = 'rs_w'
                elif abs(push_option['ovp_thr']) != 0.0:
                    push = 'overlap'
                else:
                    push = False
            else:
                push = False
        else:
            push = False
        self.push = push

    def _safe_link_or_copy(self, src_abs, dst):
        try:
            os.symlink(src_abs, dst)
            return
        except Exception as e:
            log.warning(f"Failed to create symlink {dst} -> {src_abs}, fallback to copy. Reason: {e}")
        shutil.copy2(src_abs, dst)

    def _is_dist_expert(self):
        return bool(getattr(self.trainer, "distributed_expert", False)) and dist.is_available() and dist.is_initialized()

    def _is_main(self):
        return bool(getattr(self.trainer, "is_main_process", True))

    def _to_cpu_obj(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_cpu_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_cpu_obj(v) for v in obj)
        return obj

    def _gather_dist_states(self):
        local_idx = self.trainer.local_expert_idx

        local_expert_state = self._to_cpu_obj(self.trainer.model.experts[local_idx].state_dict())

        local_opt = self.trainer.optimizers[local_idx]
        local_sch = self.trainer.lr_schedulers[local_idx]

        local_opt_state = self._to_cpu_obj(local_opt.state_dict()) if local_opt is not None else None
        local_sch_state = self._to_cpu_obj(local_sch.state_dict()) if local_sch is not None else None

        expert_states = [None for _ in range(self.trainer.world_size)]
        opt_states = [None for _ in range(self.trainer.world_size)]
        sch_states = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(expert_states, local_expert_state)
        dist.all_gather_object(opt_states, local_opt_state)
        dist.all_gather_object(sch_states, local_sch_state)

        return expert_states, opt_states, sch_states

    def _assemble_full_model_state(self, expert_states):
        base_state = self._to_cpu_obj(self.trainer.model.state_dict())
        full_state = {}

        for k, v in base_state.items():
            if not k.startswith("experts."):
                full_state[k] = v

        for i, expert_state in enumerate(expert_states):
            for k, v in expert_state.items():
                full_state[f"experts.{i}.{k}"] = v

        return full_state

    def iteration(self, **kwargs):
        if self.push == 'rs_w':
            suffix = ".iter_rs" + "%.3f" % self.trainer.model.hopping_options["rs"] + "_w" + "%.3f" % \
                     self.trainer.model.hopping_options["w"]
            max_ckpt = self.trainer.train_options["max_ckpt"]
        elif self.push == 'overlap':
            suffix = ".iter_ovp" + "%.3f" % self.trainer.model.ovp_factor
            max_ckpt = self.trainer.train_options["max_ckpt"]
        else:
            suffix = ".iter{}".format(self.trainer.iter)
            max_ckpt = self.trainer.train_options["max_ckpt"]

        name = self.trainer.model.name + suffix
        self.latest_quene.append(name)

        delete_name = None
        if len(self.latest_quene) > max_ckpt:
            delete_name = self.latest_quene.pop(0)

        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
        )

        if self._is_main():
            if delete_name is not None:
                delete_path = os.path.join(self.checkpoint_path, delete_name + ".pth")
                try:
                    os.remove(delete_path)
                except Exception:
                    log.info(f"Failed to delete the checkpoint file {delete_path}.")

            if not self.push:
                latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
                if os.path.lexists(latest_symlink):
                    os.unlink(latest_symlink)
                latest_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                latest_ckpt_abs_path = os.path.abspath(latest_ckpt)
                if not os.path.exists(latest_ckpt_abs_path):
                    raise FileNotFoundError(f"Source file {latest_ckpt_abs_path} does not exist.")
                self._safe_link_or_copy(latest_ckpt_abs_path, latest_symlink)

    def epoch(self, **kwargs):
        updated_loss = self.trainer.stats.get('validation_loss')
        if updated_loss is not None:
            updated_loss = updated_loss.get('epoch_mean', 1e6)
        else:
            updated_loss = self.trainer.stats.get("train_loss").get("epoch_mean", 1e6)

        max_ckpt = self.trainer.train_options["max_ckpt"]

        if updated_loss < self.best_loss:
            suffix = ".ep{}".format(self.trainer.ep)
            name = self.trainer.model.name + suffix
            self.best_quene.append(name)

            delete_name = None
            if len(self.best_quene) > max_ckpt:
                delete_name = self.best_quene.pop(0)

            self._save(
                name=name,
                model=self.trainer.model,
                model_options=self.trainer.model.model_options,
                common_options=self.trainer.common_options,
                train_options=self.trainer.train_options,
            )

            self.best_loss = updated_loss

            if self._is_main():
                if delete_name is not None:
                    delete_path = os.path.join(self.checkpoint_path, delete_name + ".pth")
                    if os.path.exists(delete_path):
                        os.remove(delete_path)

                best_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".best.pth")
                if os.path.lexists(best_symlink):
                    os.unlink(best_symlink)
                best_ckpt = os.path.join(self.checkpoint_path, name + ".pth")
                best_ckpt_abs_path = os.path.abspath(best_ckpt)
                if not os.path.exists(best_ckpt_abs_path):
                    raise FileNotFoundError(f"Source file {best_ckpt_abs_path} does not exist.")
                self._safe_link_or_copy(best_ckpt_abs_path, best_symlink)

    def _save(self, name, model, model_options, common_options, train_options):
        obj = {}
        obj.update({"config": {"model_options": model_options, "common_options": common_options,
                               "train_options": train_options}})

        if self._is_dist_expert():
            expert_states, opt_states, sch_states = self._gather_dist_states()
            full_model_state = self._assemble_full_model_state(expert_states)

            obj.update({
                "model_state_dict": full_model_state,
                "task": self.trainer.task,
                "epoch": self.trainer.ep,
                "iteration": self.trainer.iter,
                "stats": self.trainer.stats,
                "optimizers_state_dict": opt_states,
                "lr_schedulers_state_dict": sch_states,
            })

            if self._is_main():
                f_path = os.path.join(self.checkpoint_path, name + ".pth")
                torch.save(obj, f=f_path)
                log.info(msg="checkpoint saved as {}".format(name))

            dist.barrier()
            return

        # 单卡 / 非分布式
        if hasattr(self.trainer, "optimizers") and isinstance(self.trainer.optimizers, list):
            optim_state = {"optimizers_state_dict": [opt.state_dict() for opt in self.trainer.optimizers]}
            sched_state = {"lr_schedulers_state_dict": [sch.state_dict() for sch in self.trainer.lr_schedulers]}
        else:
            optim_state = {"optimizer_state_dict": self.trainer.optimizer.state_dict()}
            sched_state = {"lr_scheduler_state_dict": self.trainer.lr_scheduler.state_dict()}

        obj.update({
            "model_state_dict": model.state_dict(),
            "task": self.trainer.task,
            "epoch": self.trainer.ep,
            "iteration": self.trainer.iter,
            "stats": self.trainer.stats
        })

        obj.update(optim_state)
        obj.update(sched_state)

        f_path = os.path.join(self.checkpoint_path, name + ".pth")
        torch.save(obj, f=f_path)
        log.info(msg="checkpoint saved as {}".format(name))