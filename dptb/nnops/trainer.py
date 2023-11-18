import torch
import logging
from dptb.utils.tools import get_lr_scheduler, \
get_optimizer, j_must_have
from dptb.nnops.trainloss import lossfunction
from dptb.nnops.base_trainer import _BaseTrainer
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, build_dataset, AtomicData
from dptb.nn import build_model
from _loss import Loss

log = logging.getLogger(__name__)
#TODO: complete the log output for initilizing the trainer

class Trainer(_BaseTrainer):

    object_keys = ["lr_scheduler", "optimizer"]

    def __init__(
            self,
            train_options: dict,
            common_options: dict,
            model: torch.nn.Module,
            train_datasets: AtomicDataset,
            reference_datasets: Optional[AtomicDataset]=None,
            validation_datasets: Optional[AtomicDataset]=None,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            ) -> None:
        super(Trainer, self).__init__(dtype=dtype, device=device)
        
        # init the object
        self.model = model.to(device)
        self.optimizer = get_optimizer(self.model.parameters(), **train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, last_epoch=self.epoch, **self.train_options["lr_scheduler"])  # add optmizer
        
        self.train_datasets = train_datasets
        if reference_datasets is not None:
            self.reference_datesets = reference_datasets
            self.use_reference = True

        if validation_datasets is not None:
            self.validation_datasets = validation_datasets
            self.validation = True

        self.train_loader = DataLoader(dataset=self.train_datasets)

        if self.use_reference:
            self.reference_loader = DataLoader(dataset=self.reference_datesets)

        if self.validation:
            self.validation_loader = DataLoader(dataset=self.validation_datasets)

        # loss function
        self.train_lossfunc = Loss(method=train_options["loss_options"]["train"]["method"])
        if self.validation:
            self.validation_lossfunc = Loss(method=train_options["loss_options"]["validation"]["method"])

    def iteration(self, batch, ref_batch=None):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.optim.zero_grad(set_to_none=True)
        batch = batch.to(self.device)
        batch = AtomicData.to_AtomicDataDict(batch)

        batch_for_loss = batch_for_loss.copy() # make a shallow copy in case the model change the batch data
        #TODO: the rescale/normalization can be added here
        batch = self.model(batch)

        loss = self.train_lossfunc(batch, batch_for_loss)

        if ref_batch is not None:
            ref_batch = ref_batch.to(self.device)
            ref_batch = AtomicData.to_AtomicDataDict(ref_batch)
            ref_batch_for_loss = ref_batch.copy()
            ref_batch = self.model(ref_batch)
            loss += self.train_lossfunc(ref_batch, batch_for_loss)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #TODO: add clip large gradient
        self.optimizer.step()

        state = {'field':'iteration', "train_loss": loss.detach(), "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}
        self.call_plugins(queue_name='iteration', time=self.iteration, **state)
        self.iteration += 1

        #TODO: add EMA

        return loss.detach()
    
    @classmethod
    def restart(
        cls,
        checkpoint: str,
        train_datasets: AtomicDataset,
        reference_datasets: Optional[AtomicDataset]=None,
        validation_datasets: Optional[AtomicDataset]=None,
        ):
        """init trainer from disk"""

        ckpt = torch.load(checkpoint)

        model = build_model(**ckpt["config"]["model_options"], **ckpt["config"]["common_options"])

        # init trainer and load the trainer's states
        trainer = cls(
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
            train_options=ckpt["config"]["train_options"],
            common_options=ckpt["config"]["common_options"],
            dtype=ckpt["config"]["common_options"]["dtype"],
            device=ckpt["config"]["common_options"]["device"],
            )
        
        trainer.epoch = ckpt["epoch"]
        trainer.iteration = ckpt["iteration"]
        trainer.stats = ckpt["stats"]

        queues_name = list(trainer.plugin_queues.keys())
        for unit in queues_name:
            for plugin in trainer.plugin_queues[unit]:
                plugin = (getattr(trainer, unit) + plugin[0], plugin[1], plugin[2])

        for key in Trainer.object_keys:
            item = getattr(trainer, key, None)
            if item is not None:
                item.load_state_dict(checkpoint[key+"state_dict"])
# 

    def epoch(self) -> None:

        for ibatch in self.train_loader:
            # iter with different structure
            if self.use_reference:
                self.iteration(ibatch, next(self.reference_loader))
            else:
                self.iteration(ibatch)


    def update(self, **kwargs):
        pass

    def validation(self, fast=True):
        with torch.zero_grad():
            loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)

            for ibatch in self.validation_loader:
                batch = batch.to(self.device)
                batch = AtomicData.to_AtomicDataDict(batch)

                batch_for_loss = batch_for_loss.copy()
                batch = self.model(batch)

                loss += self.validation_lossfunc(batch, batch_for_loss)

                if fast:
                    break

        return loss
