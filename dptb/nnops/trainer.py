import torch
import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from future.utils import with_metaclass
from dptb.utils.constants import dtype_dict
from dptb.plugins.base_plugin import PluginUser


log = logging.getLogger(__name__)


class Trainer(with_metaclass(ABCMeta, PluginUser)):

    def __init__(self, jdata) -> None:
        super(Trainer, self).__init__()
        self.dtype = dtype_dict[jdata.get("dtype", "float32")]
        self.device = jdata.get("device", "cpu")
        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.  
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training. 
                        - batch: events before batch training. 
                        - epoch: events after epoch batch training 
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.iteration = 1
        self.epoch = 1

    

    @abstractmethod
    def _init_param(self, jdata):

        pass

    @abstractmethod
    def _init_model(self):
        '''
        init the model
        '''
        pass

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train()
            # run plugins of epoch events.
            self.call_plugins(queue_name='epoch', time=i)
            self.lr_scheduler.step()  # modify the lr at each epoch (should we add it to pluggins so we could record the lr scheduler process?)
            self.epoch += 1


    @abstractmethod
    def calc(self, **data):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def validation(self, **kwargs):
        pass



if __name__ == '__main__':
    a = [1, 2, 3]

    print(list(enumerate(a, 2)))