import torch
import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from future.utils import with_metaclass
from dptb.utils.constants import dtype_dict


log = logging.getLogger(__name__)


class Trainer(with_metaclass(ABCMeta, object)):

    def __init__(self, jdata) -> None:
        self.dtype = dtype_dict(jdata.get("dtype", "float32"))
        self.device = jdata.get("device", "cpu")

        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.  
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training. 
                        - batch: events before batch training. 
                        - epoch: events after epoch batch training 
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.stats = {}  # the status of Trainer.
        self.plugin_queues = {'iteration': [], 'epoch': [], 'batch': [], 'update': []}
        self.iteration = 1

    def _check_param(self, jdata):
        pass


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

    def register_plugin(self, plugin):
        plugin.register(self)

        # the trigger interval of plugin, with the form like: [(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            # unit the plugin type.
            queue = self.plugin_queues[unit]
            # Add the plugin events. duration is the trigger interval. len(queue) is the priority levels for the same duration,
            # the smaller the higher and is determined by the order of registration.
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, **kwargs):
        # args should contain: [input, target, output, loss]
        kwargs.update({"time": time})
        # time can be iteration or epoch ...
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            # the plugin must have at-least one of the iteration、batch、epoch and update events.
            getattr(plugin, queue_name)(**kwargs)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            # 根据插件的事件触发间隔，来更新事件队列里的事件 duration
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)
            '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''

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