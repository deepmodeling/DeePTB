from dptb.utils.base_plugin import Plugin
from collections import defaultdict
import logging
import os
import time
import torch

log = logging.getLogger(__name__)

class Saver(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(10, 'iteration'), (1, 'epoch')]
        super(Saver, self).__init__(interval)
        self.best_loss = 1e7

    def register(self, trainer):
        self.checkpoint_path = trainer.run_opt["checkpoint_path"]
        self.trainer = trainer

    def iteration(self, **kwargs):
        self._save("latest_"+self.trainer.name)

    def epoch(self, **kwargs):
        if self.trainer.stats.get('validation_loss').get('last',1e6) < self.best_loss:
            self._save("best_"+self.trainer.name)
            self.best_loss = self.trainer.stats['validation_loss'].get('last',1e6)

            log.info(msg="checkpoint saved as {}".format("best_epoch"))

    def _save(self, name):
        obj = {}
        obj.update({"model_config":self.trainer.model_config, "state_dict":self.trainer.model.state_dict(), "sk_options":self.trainer.sk_options})
        f_path = os.path.join(self.checkpoint_path, name+".pth")
        torch.save(obj, f=f_path)

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
