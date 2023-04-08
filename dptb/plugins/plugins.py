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

    def register(self, trainer):
        self.checkpoint_path = trainer.run_opt["checkpoint_path"]
        self.trainer = trainer

    def iteration(self, **kwargs):
        suffix = "_c"+str(self.trainer.model_options["skfunction"]["sk_cutoff"])+"w"+str(self.trainer.model_options["skfunction"]["sk_decay_w"])
        self._save(name="latest_"+self.trainer.name+suffix,model=self.trainer.model,model_config=self.trainer.model_config)
        if self.trainer.name == "dptb" \
                and self.trainer.run_opt["use_correction"] \
                    and not self.trainer.run_opt["freeze"]:

            self._save(name="latest_"+self.trainer.name+'_nnsk_'+suffix,model=self.trainer.sknet, model_config=self.trainer.sknet_config)

    def epoch(self, **kwargs):
        if self.trainer.stats.get('validation_loss').get('last',1e6) < self.best_loss:
            suffix = "_c"+str(self.trainer.model_options["skfunction"]["sk_cutoff"])+"w"+str(self.trainer.model_options["skfunction"]["sk_decay_w"])
            self._save(name="best_"+self.trainer.name+suffix,model=self.trainer.model,model_config=self.trainer.model_config)
            self.best_loss = self.trainer.stats['validation_loss'].get('last',1e6)

            if self.trainer.name == "dptb" \
                and self.trainer.run_opt["use_correction"] \
                    and not self.trainer.run_opt["freeze"]:

                self._save(name="best_"+self.trainer.name+'_nnsk_'+suffix,model=self.trainer.sknet, model_config=self.trainer.sknet_config)

            # log.info(msg="checkpoint saved as {}".format("best_epoch"))

    def _save(self, name, model, model_config):
        obj = {}
        model_config["dtype"] = str(model_config["dtype"]).split('.')[-1]
        obj.update({"model_config":model_config, "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": self.trainer.optimizer.state_dict(), "epoch": self.trainer.epoch+1, 
        "iteration":self.trainer.iteration+1, "stats": self.trainer.stats})
        f_path = os.path.join(self.checkpoint_path, name+".pth")
        torch.save(obj, f=f_path)

        # json_model_types = ["onsite", "hopping","soc"]
        if  self.trainer.name == "nnsk":
            json_data = {}
            onsitecoeff = {}
            hoppingcoeff = {}
            if self.trainer.onsitemode ==  "strain":
                for i in self.trainer.onsite_coeff:
                    onsitecoeff[i] = self.trainer.onsite_coeff[i].tolist()
            elif self.trainer.onsitemode in ['uniform','split']:
                for ia in self.trainer.onsite_coeff:
                    for iikey in range(len(self.trainer.onsite_index_dict[ia])):
                        onsitecoeff[self.trainer.onsite_index_dict[ia][iikey]] = \
                                        [self.trainer.onsite_coeff[ia].tolist()[iikey]]

            json_data["onsite"] = onsitecoeff
            for i in self.trainer.hopping_coeff:
                hoppingcoeff[i] = self.trainer.hopping_coeff[i].tolist()
            json_data["hopping"] = hoppingcoeff
            if hasattr(self.trainer,'soc_coeff'):
                soccoeff = {}
                for ia in self.trainer.soc_coeff:
                    for iikey in range(len(self.trainer.onsite_index_dict[ia])):
                        soccoeff[self.trainer.onsite_index_dict[ia][iikey]] = \
                            [self.trainer.soc_coeff[ia].tolist()[iikey]]
                json_data["soc"] = soccoeff
            json_path = os.path.join(self.checkpoint_path, name+".json")
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)
            
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
