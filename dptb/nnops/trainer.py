import torch
import logging
from dptb.utils.tools import get_lr_scheduler, \
get_optimizer, j_must_have
from dptb.nnops.trainloss import lossfunction
from dptb.nnops.base_trainer import _BaseTrainer
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, build_dataset
from dptb.nn import build_model

log = logging.getLogger(__name__)

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
        self.name = "dptb"
        
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
        self.train_lossfunc = None
        self.validation_lossfunc = None

    def calc(self, batch):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''

        batch = self.model(batch)
    
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

    def train(self) -> None:

        for ibatch in self.dataloader:
            # iter with different structure

            def closure():
                # calculate eigenvalues.
                self.optimizer.zero_grad()
                ibatch = self.calc(ibatch)

                loss = self.train_lossfunc(ibatch, **self.loss_options)

                if self.use_reference:
                    for irefbatch in range(self.ref_loader):
                        irefbatch = self.calc(irefbatch)
                        loss += (self.batch_size * 1.0 / (self.reference_batch_size * (1+self.n_reference_sets))) * \
                                    self.train_lossfunc(ibatch, **self.reference_loss_options)
                
                loss.backward()
                self.train_loss = loss.detach()
                return loss

            self.optimizer.step(closure)
            state = {'field':'iteration', "train_loss": self.train_loss, "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

            self.call_plugins(queue_name='iteration', time=self.iteration, **state)
            # self.lr_scheduler.step() # 在epoch 加入 scheduler.
            self.iteration += 1

    def update(self, **kwargs):
        pass

    def validation(self, quick=False):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            for processor in self.validation_processor_list:
                self.validation_loss_options.update(processor.bandinfo)
                for data in processor:
                    eigenvalues_pred, eigenvalues_lbl = self.calc(*data)
                    total_loss += self.validation_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.validation_loss_options)
                    if quick:
                        break
                    
        with torch.enable_grad():
            return total_loss.detach()



if __name__ == '__main__':
    a = [1,2,3]

    print(list(enumerate(a, 2)))
