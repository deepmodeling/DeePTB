import torch
import logging
from dptb.utils.tools import get_lr_scheduler, \
get_optimizer, j_must_have
from dptb.nnops.base_tester import BaseTester
from typing import Union, Optional
from dptb.data import AtomicDataset, DataLoader, AtomicData
from dptb.nn import build_model
from dptb.nnops.loss import Loss

log = logging.getLogger(__name__)
#TODO: complete the log output for initilizing the trainer

class Tester(BaseTester):

    def __init__(
            self,
            test_options: dict,
            common_options: dict,
            model: torch.nn.Module,
            test_datasets: AtomicDataset,
            ) -> None:
        super(Tester, self).__init__(dtype=common_options["dtype"], device=common_options["device"])
        
        # init the object
        self.model = model.to(self.device)
        self.common_options = common_options
        self.test_options = test_options
        
        self.test_datasets = test_datasets

        self.test_loader = DataLoader(dataset=self.train_datasets, batch_size=test_options["batch_size"], shuffle=False)

        # loss function
        self.test_lossfunc = Loss(**test_options["loss_options"]["test"], **common_options, idp=self.model.hamiltonian.idp)
    
    def iteration(self, batch):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        self.model.eval()
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
        #TODO: the rescale/normalization can be added here
        batch = self.model(batch)

        #TODO: this could make the loss function unjitable since t he batchinfo in batch and batch_for_loss does not necessarily 
        #       match the torch.Tensor requiresment, should be improved further

        batch.update(batch_info)
        batch_for_loss.update(batch_info)

        loss = self.train_lossfunc(batch, batch_for_loss)

        state = {'field':'iteration', "test_loss": loss.detach()}
        self.call_plugins(queue_name='iteration', time=self.iter, **state)
        self.iter += 1

        return loss.detach()
    
    def epoch(self) -> None:

        for ibatch in self.test_loader:
            # iter with different structure
            self.iteration(ibatch)