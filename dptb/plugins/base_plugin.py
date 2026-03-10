import heapq
import logging
from dptb.utils.tools import get_lr_scheduler, j_must_have, get_optimizer
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from future.utils import with_metaclass
from dptb.utils.constants import dtype_dict

class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, *args):
        raise NotImplementedError

class PluginUser(object):
    def __init__(self) -> None:
        ''' Here is for plugins.
                    plugins:
                        - iteration: events  after every batch training iteration.  
                        - update: the updates of model paras including networks and optimiser, such as leaning rate, etc. after the batch training. 
                        - batch: events before batch training. 
                        - epoch: events after epoch batch training 
                    The difference b/w iteration and update the parameters, iteration takes in the batch output, loss etc., while  update takes in model itself.
                '''
        self.stats = {}  # the status of Trainer.
        self.plugin_queues = {'disposable': [], 'iteration': [], 'epoch': [], 'batch': [], 'update': []}

    def register_plugin(self, plugin, **kwargs):
        plugin.register(self, **kwargs)

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
        # TODO: why we need a time update here?
        # kwargs.update({"time": time})
        # time can be iteration or epoch ...
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            # the plugin must have at-least one of the iteration、batch、epoch and update events.
            getattr(plugin, queue_name)(time=time, **kwargs)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            # 根据插件的事件触发间隔，来更新事件队列里的事件 duration
            if queue[0][0] > 0:
                new_item = (time + interval, queue[0][1], plugin)
                heapq.heappushpop(queue, new_item)
                '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''
            else:
                heapq.heappop(queue)
                if len(queue) == 0:
                    return
