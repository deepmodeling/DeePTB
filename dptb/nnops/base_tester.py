import torch
import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from future.utils import with_metaclass
from typing import Union
from dptb.utils.constants import dtype_dict
from dptb.plugins.base_plugin import PluginUser


log = logging.getLogger(__name__)


class Tester(with_metaclass(ABCMeta, PluginUser)):

    def __init__(self, jdata) -> None:
        super(Tester, self).__init__()
        self.dtype = jdata["common_options"]["dtype"]
        self.device = jdata["common_options"]["device"]
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
    def build(self):
        '''
        init the model
        '''
        pass

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1):
            self.test()
            # run plugins of epoch events.
            self.call_plugins(queue_name='epoch', time=i)
            self.epoch += 1

    @abstractmethod
    def calc(self, **data):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        pass

    @abstractmethod
    def test(self) -> None:
        pass

class BaseTester(with_metaclass(ABCMeta, PluginUser)):

    def __init__(
            self, 
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            ) -> None:
        super(BaseTester, self).__init__()

        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]
        self.dtype = dtype
        self.device = device

        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.  
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training. 
                        - batch: events before batch training. 
                        - epoch: events after epoch batch training 
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.iter = 1
        self.ep = 1

    def run(self):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        self.epoch()
        # run plugins of epoch events.
        self.call_plugins(queue_name='epoch', time=i)
        self.lr_scheduler.step()  # modify the lr at each epoch (should we add it to pluggins so we could record the lr scheduler process?)
        self.ep += 1


    @abstractmethod
    def iteration(self, **data):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        pass

    @abstractmethod
    def epoch(self) -> None:
        """define a training iteration process
        """
        pass


if __name__ == '__main__':
    a = [1, 2, 3]

    print(list(enumerate(a, 2)))