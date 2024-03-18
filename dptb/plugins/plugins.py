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

    def iteration(self, **kwargs):
        # suffix = "_b"+"%.3f"%self.trainer.common_options["bond_cutoff"]+"_c"+"%.3f"%self.trainer.onsite_options["skfunction"]["sk_cutoff"]+"_w"+\
        #         "%.3f"%self.trainer.model_options["skfunction"]["sk_decay_w"]
        suffix = ".iter{}".format(self.trainer.iter)
        name = self.trainer.model.name+suffix
        self.latest_quene.append(name)
        if len(self.latest_quene) >= 5:
            delete_name = self.latest_quene.pop(0)
            delete_path = os.path.join(self.checkpoint_path, delete_name+".pth")
            os.remove(delete_path)

        self._save(
            name=name,
            model=self.trainer.model,
            model_options=self.trainer.model.model_options,
            common_options=self.trainer.common_options,
            train_options=self.trainer.train_options,
            )
        
        # if self.trainer.name == "dptb" \
        #         and self.trainer.run_opt["use_correction"] \
        #             and not self.trainer.run_opt["freeze"]:

        #     self._save(name="latest_"+self.trainer.name+'_nnsk'+suffix,model=self.trainer.sknet, model_config=self.trainer.sknet_config)

    def epoch(self, **kwargs):

        updated_loss = self.trainer.stats.get('validation_loss')
        if updated_loss is not None:
            updated_loss = updated_loss.get('epoch_mean',1e6)
        else:
            updated_loss = self.trainer.stats.get("train_loss").get("epoch_mean",1e6)


        if updated_loss < self.best_loss:
            # suffix = "_b"+"%.3f"%self.trainer.common_options["bond_cutoff"]+"_c"+"%.3f"%self.trainer.model_options["skfunction"]["sk_cutoff"]+"_w"+\
            #     "%.3f"%self.trainer.model_options["skfunction"]["sk_decay_w"]
            suffix = ".ep{}".format(self.trainer.ep)
            name = self.trainer.model.name+suffix
            self.best_quene.append(name)
            if len(self.best_quene) >= 5:
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

            # if self.trainer.name == "dptb" \
            #     and self.trainer.run_opt["use_correction"] \
            #         and not self.trainer.run_opt["freeze"]:

            #     self._save(
            #         name="best_"+self.trainer.name+'_nnsk'+suffix,
            #         model=self.trainer.sknet, 
            #         model_config=self.trainer.sknet_config
            #         common_options=self.trainer.common_options
            #         )

            # log.info(msg="checkpoint saved as {}".format("best_epoch"))

    def _save(self, name, model, model_options, common_options, train_options):
        obj = {}
        obj.update({"config": {"model_options": model_options, "common_options": common_options, "train_options": train_options}})
        obj.update(
            {
                "model_state_dict": model.state_dict(), 
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

class Validationer(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Validationer, self).__init__(interval)

    def register(self, trainer):
        self.trainer = trainer
        validation_stats = self.trainer.stats.setdefault('validation_loss', {})
        validation_stats['log_epoch_fields'] = ['{last:.4f}']
        validation_stats['log_iter_fields'] = ['{last:.4f}']

    def epoch(self, **kwargs):
        self.trainer.model.eval()

        validation_stats = self.trainer.stats.setdefault('validation_loss', {})
        validation_stats['last'] = self.trainer.validation()

        self.trainer.model.train()

    def iteration(self, **kwargs):
        self.trainer.model.eval()

        validation_stats = self.trainer.stats.setdefault('validation_loss', {})
        validation_stats['last'] = self.trainer.validation(quick=True)

        self.trainer.model.train()
