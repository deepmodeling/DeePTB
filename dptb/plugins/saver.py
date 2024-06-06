from dptb.plugins.base_plugin import Plugin
from collections import defaultdict
import logging
import os
import time
import torch
import json

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
            # 获取 push 选项
            push_option = self.trainer.model.model_options["nnsk"].get("push", False)
            if push_option:
                # 计算所有阈值之和
                thrs = sum(abs(val) for key, val in push_option.items() if "thr" in key)
                # 如果阈值之和不为 0, 则 push 为 True
                if abs(push_option['rs_thr'])  + abs(push_option['w_thr']) != 0.0 and abs(push_option['ovp_thr']) != 0.0:
                    log.error("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")
                    raise ValueError("rs_thr, w_thr and ovp_thr cannot be pushed at the same time.")
                
                if abs(push_option['rs_thr'])  + abs(push_option['w_thr']) != 0.0:
                    push = 'rs_w'
                # push = abs(thrs) != 0.0
                elif abs(push_option['ovp_thr']) != 0.0:
                    push = 'overlap'
                else:
                    push = False            
            else:
                push = False
        else:
            push = False
        self.push = push   
        

    def iteration(self, **kwargs):
        # suffix = "_b"+"%.3f"%self.trainer.common_options["bond_cutoff"]+"_c"+"%.3f"%self.trainer.onsite_options["skfunction"]["sk_cutoff"]+"_w"+\
        #         "%.3f"%self.trainer.model_options["skfunction"]["sk_decay_w"]
        if self.push == 'rs_w':
            suffix = ".iter_rs" + "%.3f"%self.trainer.model.hopping_options["rs"]+"_w"+"%.3f"%self.trainer.model.hopping_options["w"]
            # By default, the maximum number of saved checkpoints is 100 for pushing rs and w.
            max_ckpt = self.trainer.train_options["max_ckpt"]
        elif self.push == 'overlap':
            suffix= ".iter_ovp" + "%.3f"%self.trainer.model.ovp_factor
            max_ckpt = self.trainer.train_options["max_ckpt"]
        else:
            suffix = ".iter{}".format(self.trainer.iter)
            max_ckpt = self.trainer.train_options["max_ckpt"]

        name = self.trainer.model.name+suffix
        self.latest_quene.append(name)
        
        if len(self.latest_quene) > max_ckpt:
            delete_name = self.latest_quene.pop(0)
            delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
            os.remove(delete_path)

        if len(self.latest_quene) > max_ckpt:
            delete_name = self.latest_quene.pop(0)
            delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
            try:        
                os.remove(delete_path)
            except:
                log.info(f"Failed to delete the checkpoint file {delete_path}.")
                
        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
            )
        
        if not self.push:
        # 构建一个符号链接，指向最新的模型
            latest_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".latest.pth")
            if os.path.lexists(latest_symlink):
                os.unlink(latest_symlink)
            latest_ckpt = os.path.join(self.checkpoint_path, name+".pth")
            latest_ckpt_abs_path = os.path.abspath(latest_ckpt)
            # 确保源文件存在
            if not os.path.exists(latest_ckpt_abs_path):
                raise FileNotFoundError(f"Source file {latest_ckpt_abs_path} does not exist.")
            os.symlink(latest_ckpt_abs_path, latest_symlink)

    def epoch(self, **kwargs):

        updated_loss = self.trainer.stats.get('validation_loss')
        if updated_loss is not None:
            updated_loss = updated_loss.get('epoch_mean',1e6)
        else:
            updated_loss = self.trainer.stats.get("train_loss").get("epoch_mean",1e6)

        max_ckpt = self.trainer.train_options["max_ckpt"]

        if updated_loss < self.best_loss:
            # suffix = "_b"+"%.3f"%self.trainer.common_options["bond_cutoff"]+"_c"+"%.3f"%self.trainer.model_options["skfunction"]["sk_cutoff"]+"_w"+\
            #     "%.3f"%self.trainer.model_options["skfunction"]["sk_decay_w"]
            suffix = ".ep{}".format(self.trainer.ep)
            name = self.trainer.model.name+suffix
            self.best_quene.append(name)
            if len(self.best_quene) > max_ckpt:
                delete_name = self.best_quene.pop(0)
                delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
                os.remove(delete_path)

            self._save(
                name=name,
                model=self.trainer.model,
                model_options=self.trainer.model.model_options,
                common_options=self.trainer.common_options,
                train_options=self.trainer.train_options,
                )
            
            self.best_loss = updated_loss

            # 构建一个符号链接，指向best模型
            best_symlink = os.path.join(self.checkpoint_path, self.trainer.model.name + ".best.pth")
            if os.path.lexists(best_symlink):
                os.unlink(best_symlink)
            best_ckpt = os.path.join(self.checkpoint_path, name+".pth")
            best_ckpt_abs_path = os.path.abspath(best_ckpt)
            # 确保源文件存在
            if not os.path.exists(best_ckpt_abs_path):
                raise FileNotFoundError(f"Source file {best_ckpt_abs_path} does not exist.")
            os.symlink(best_ckpt_abs_path, best_symlink)

    def _save(self, name, model, model_options, common_options, train_options):
        obj = {}
        obj.update({"config": {"model_options": model_options, "common_options": common_options, "train_options": train_options}})
        obj.update(
            {
                "model_state_dict": model.state_dict(),
                "task": self.trainer.task,
                "optimizer_state_dict": self.trainer.optimizer.state_dict(), 
                "lr_scheduler_state_dict": self.trainer.lr_scheduler.state_dict(),
                "epoch": self.trainer.ep,
                "iteration":self.trainer.iter, 
                "stats": self.trainer.stats}
                )
        f_path = os.path.join(self.checkpoint_path, name+".pth")
        torch.save(obj, f=f_path)

        # # json_model_types = ["onsite", "hopping","soc"]
        # if  self.trainer.name == "nnsk":
        #     json_data = {}
        #     onsitecoeff = {}
        #     hoppingcoeff = {}
        #     if self.trainer.onsitemode ==  "strain":
        #         for i in self.trainer.onsite_coeff:
        #             onsitecoeff[i] = self.trainer.onsite_coeff[i].tolist()
        #     elif self.trainer.onsitemode in ['uniform','split']:
        #         for ia in self.trainer.onsite_coeff:
        #             for iikey in range(len(self.trainer.onsite_index_dict[ia])):
        #                 onsitecoeff[self.trainer.onsite_index_dict[ia][iikey]] = \
        #                                 [self.trainer.onsite_coeff[ia].tolist()[iikey]]
        #     elif self.trainer.onsitemode == 'NRL':
        #         for i in self.trainer.onsite_coeff:
        #             onsitecoeff[i] = self.trainer.onsite_coeff[i].tolist()   
            
        #     json_data["onsite"] = onsitecoeff
        #     for i in self.trainer.hopping_coeff:
        #         hoppingcoeff[i] = self.trainer.hopping_coeff[i].tolist()
        #     json_data["hopping"] = hoppingcoeff

        #     if self.trainer.overlap_coeff is not None:
        #         overlapcoeff = {}
        #         for i in self.trainer.overlap_coeff:
        #             overlapcoeff[i] = self.trainer.overlap_coeff[i].tolist()
        #         json_data["overlap"] = overlapcoeff
                
        #     if hasattr(self.trainer,'soc_coeff'):
        #         soccoeff = {}
        #         for ia in self.trainer.soc_coeff:
        #             for iikey in range(len(self.trainer.onsite_index_dict[ia])):
        #                 soccoeff[self.trainer.onsite_index_dict[ia][iikey]] = \
        #                     [self.trainer.soc_coeff[ia].tolist()[iikey]]
        #         json_data["soc"] = soccoeff
        #     json_path = os.path.join(self.checkpoint_path, name+".json")
        #     with open(json_path, "w") as f:
        #         json.dump(json_data, f, indent=4)
            
        log.info(msg="checkpoint saved as {}".format(name))
