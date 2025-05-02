import torch
import logging
from dptb.utils.tools import get_lr_scheduler, \
get_optimizer, j_must_have
from dptb.nnops.base_trainer import BaseTrainer
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, AtomicData
from dptb.nn import build_model
from dptb.nnops.loss import Loss

log = logging.getLogger(__name__)
#TODO: complete the log output for initilizing the trainer

class Trainer(BaseTrainer):

    object_keys = ["lr_scheduler", "optimizer"]

    def __init__(
            self,
            train_options: dict,
            common_options: dict,
            model: torch.nn.Module,
            train_datasets: AtomicDataset,
            reference_datasets: Union[AtomicDataset, None]=None,
            validation_datasets: Union[AtomicDataset, None]=None,
            ) -> None:
        super(Trainer, self).__init__(dtype=common_options["dtype"], device=common_options["device"])
        
        # init the object
        self.model = model.to(self.device)
        self.optimizer = get_optimizer(model_param=self.model.parameters(), **train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **train_options["lr_scheduler"])  # add optmizer
        self.update_lr_per_step_flag = train_options["update_lr_per_step_flag"]
        self.common_options = common_options
        self.train_options = train_options
        
        self.train_datasets = train_datasets
        # get the task from train_datasets label
        self.task = None
        if self.train_datasets.get_Hamiltonian:
            self.task = "hamiltonians"
        elif self.train_datasets.get_DM:
            self.task = "DM"
        else:
            self.task = "eigenvalues"

        self.use_reference = False
        if reference_datasets is not None:
            self.reference_datesets = reference_datasets
            self.use_reference = True

        if validation_datasets is not None:
            self.validation_datasets = validation_datasets
            self.use_validation = True
        else:
            self.use_validation = False

        self.train_loader = DataLoader(dataset=self.train_datasets, batch_size=train_options["batch_size"], shuffle=True)

        if self.use_reference:
            self.reference_loader = DataLoader(dataset=self.reference_datesets, batch_size=train_options["ref_batch_size"], shuffle=True)

        if self.use_validation:
            self.validation_loader = DataLoader(dataset=self.validation_datasets, batch_size=train_options["val_batch_size"], shuffle=True)

        # loss function
        self.train_lossfunc = Loss(**train_options["loss_options"]["train"], **common_options, idp=self.model.hamiltonian.idp)
        if self.use_validation:
            self.validation_lossfunc = Loss(**train_options["loss_options"]["validation"], **common_options, idp=self.model.hamiltonian.idp)
        if self.use_reference:
            self.reference_lossfunc = Loss(**train_options["loss_options"]["reference"], **common_options, idp=self.model.hamiltonian.idp)

        if  train_options["loss_options"]["train"]["method"] == "skints":
            assert self.model.name == 'nnsk', "The model should be nnsk for the skints loss function."
            assert self.model.onsite_fn.functype in ['none', 'uniform'], "The onsite function should be none or uniform for the skints loss function."
            log.info("The skints loss function is used for training, the model.transform is then set to False.")
            self.model.transform = False

    def iteration(self, batch, ref_batch=None):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = batch.to(self.device)
        
        # record the batch_info to help reconstructing sub-graph from the batch
        batch_info = {
            "__slices__": batch.__slices__,
            "__cumsum__": batch.__cumsum__,
            "__cat_dims__": batch.__cat_dims__,
            "__num_nodes_list__": batch.__num_nodes_list__,
            "__data_class__": batch.__data_class__,
        }

        batch = AtomicData.to_AtomicDataDict(batch)

        batch_for_loss = batch.copy() # make a shallow copy in case the model change the batch data
        
        batch = self.model(batch)

        #TODO: this could make the loss function unjitable since t he batchinfo in batch and batch_for_loss does not necessarily 
        #       match the torch.Tensor requiresment, should be improved further

        batch.update(batch_info)
        batch_for_loss.update(batch_info)

        loss = self.train_lossfunc(batch, batch_for_loss)

        if ref_batch is not None:
            ref_batch = ref_batch.to(self.device) # AtomicData Type
            batch_info = {
                "__slices__": ref_batch.__slices__,
                "__cumsum__": ref_batch.__cumsum__,
                "__cat_dims__": ref_batch.__cat_dims__,
                "__num_nodes_list__": ref_batch.__num_nodes_list__,
                "__data_class__": ref_batch.__data_class__,
            }

            ref_batch = AtomicData.to_AtomicDataDict(ref_batch) # AtomicDataDict Type
            ref_batch_for_loss = ref_batch.copy()
            ref_batch = self.model(ref_batch)

            ref_batch.update(batch_info)
            ref_batch_for_loss.update(batch_info)
            
            loss += self.train_lossfunc(ref_batch, ref_batch_for_loss)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #TODO: add clip large gradient
        self.optimizer.step()
        if self.update_lr_per_step_flag:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.stats["train_loss"]["epoch_mean"])
            else:
                self.lr_scheduler.step()

        state = {'field':'iteration', "train_loss": loss.detach(), "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}
        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        #TODO: add EMA

        return loss.detach()
    
    @classmethod
    def restart(
        cls,
        checkpoint: str,
        train_datasets: AtomicDataset,
        train_options: dict={},
        common_options: dict={},
        reference_datasets: Optional[AtomicDataset]=None,
        validation_datasets: Optional[AtomicDataset]=None,
        ):
        """restart the training from a checkpoint, it does not support model options change."""

        ckpt = torch.load(checkpoint, map_location=common_options["device"], weights_only=False)
        model = build_model(checkpoint, ckpt["config"]["model_options"], ckpt["config"]["common_options"])
        if len(train_options) == 0:
            train_options = ckpt["config"]["train_options"]
        if len(common_options) == 0:
            common_options = ckpt["config"]["common_options"]

        # init trainer and load the trainer's states
        trainer = cls(
            model=model,
            train_datasets=train_datasets,
            reference_datasets=reference_datasets,
            validation_datasets=validation_datasets,
            train_options=train_options,
            common_options=common_options,
            )
        
        trainer.ep = ckpt["epoch"] + 1
        trainer.iter = ckpt["iteration"] + 1
        trainer.stats = ckpt["stats"]

        queues_name = list(trainer.plugin_queues.keys())
        for unit in queues_name:
            for plugin in trainer.plugin_queues[unit]:
                plugin = (getattr(trainer, unit) + plugin[0], plugin[1], plugin[2])

        for key in Trainer.object_keys:
            item = getattr(trainer, key, None)
            if item is not None:
                item.load_state_dict(ckpt[key+"_state_dict"])

        return trainer
# 

    def epoch(self) -> None:

        for ibatch in self.train_loader:
            # iter with different structure
            if self.use_reference:
                self.iteration(ibatch, next(iter(self.reference_loader)))
            else:
                self.iteration(ibatch)

    def update(self, **kwargs):
        pass

    def validation(self, fast=True):
        with torch.no_grad():
            loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            self.model.eval()

            for batch in self.validation_loader:
                batch = batch.to(self.device)

                batch_info = {
                    "__slices__": batch.__slices__,
                    "__cumsum__": batch.__cumsum__,
                    "__cat_dims__": batch.__cat_dims__,
                    "__num_nodes_list__": batch.__num_nodes_list__,
                    "__data_class__": batch.__data_class__,
                }

                batch = AtomicData.to_AtomicDataDict(batch)

                batch_for_loss = batch.copy()
                batch = self.model(batch)

                batch.update(batch_info)
                batch_for_loss.update(batch_info)

                loss += self.validation_lossfunc(batch, batch_for_loss)

                if fast:
                    break

        return loss
