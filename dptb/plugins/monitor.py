import logging
import time

import torch
from dptb.data import AtomicData
from dptb.plugins.base_plugin import Plugin
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

# ToDo: modify the monitor for our model.

class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):

        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        #规定了输出格式
        number_format = ':' + number_format
        # interval for initialize the base plugin 
        # list[(1, 'iteration'), (1, 'epoch')] is to define the trigger_interval.
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        # smoothing value for calculate running average.
        self.smoothing = smoothing
        # if cal running average 
        self.with_running_average = running_average
        # if cal epoch average
        self.with_epoch_average = epoch_average

        #输出日志的格式
        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        # `stat_name` need to be defined in the specific plugin.
        # setdefault returns the value in the dictionary if it exists,
        # otherwise it creates a new one with the input value as defult.
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, **kwargs):
        '''`iteration` is called at the end of each iteration, and it updates the `stats` dictionary with the
        current value of `stats['last']`
        
        '''

        # `stat_name` need to be defined in the specific plugin.
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # update the value of `stats['last']`
        # The _get_value method must be defined in the specific plugin.!。
        stats['last'] = self._get_value(**kwargs)

        if self.with_epoch_average:
            # return (sum_stats['last'], count)
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                stats['last'] * (1 - self.smoothing)

    def epoch(self, **kwargs):
        '''It computes the average of the values of epoch_stats key in the `stats` dictionary, 
        and then resets the value in the dictionary to `(0, 0)`.
        
        Parameters
        ----------
        idx
            the index of the epoch
        
        '''

        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)


class TrainLossMonitor(Monitor):
    # It's a Monitor that records the loss.
    # stat_name is used in the Monitor class to register.
    stat_name = 'train_loss'
    def __init__(self):
        super(TrainLossMonitor, self).__init__(
            precision=6,
        )
    def _get_value(self, **kwargs):
        return kwargs.get('train_loss', None)

class TestLossMonitor(Monitor):
    # It's a Monitor that records the loss.
    # stat_name is used in the Monitor class to register.
    stat_name = 'test_loss'
    def __init__(self):
        super(TestLossMonitor, self).__init__(
            precision=6,
        )

    def _get_value(self, **kwargs):
        return kwargs.get('test_loss', None)
class LearningRateMonitor(Monitor):
    # It's a Monitor that records the loss.
    # stat_name is used in the Monitor class to register.
    stat_name = 'lr'
    def __init__(self):
        super(LearningRateMonitor, self).__init__(
            running_average=False, epoch_average=False, smoothing=0.7,
            precision=6, number_format='.{}g'.format(4), unit=''
        )
    def _get_value(self, **kwargs):
        return kwargs.get('lr', None)


class Validationer(Monitor):
    stat_name = 'validation_loss'
    def __init__(self):
        super(Validationer, self).__init__(
            precision=6,
        )

    def _get_value(self, **kwargs):
        if kwargs.get('field') == "iteration":
            return self.trainer.validation(fast=True)
        else:
            return self.trainer.validation()
        

class TensorBoardMonitor(Plugin):
    def __init__(self, interval):
        super(TensorBoardMonitor, self).__init__(interval=interval)
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, **kwargs):
        epoch = self.trainer.ep
        self.writer.add_scalar(f'lr/epoch', self.trainer.stats['lr']['last'], epoch)
        self.writer.add_scalar(f'train_loss_last/epoch', self.trainer.stats['train_loss']['last'], epoch)
        self.writer.add_scalar(f'train_loss_mean/epoch', self.trainer.stats['train_loss']['epoch_mean'], epoch)

    def iteration(self, **kwargs):
        iteration = self.trainer.iter
        self.writer.add_scalar(f'lr_iter/iteration', self.trainer.stats['lr']['last'], iteration)
        self.writer.add_scalar(f'train_loss_iter/iteration', self.trainer.stats['train_loss']['last'], iteration)
