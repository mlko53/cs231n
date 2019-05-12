import numpy as np
import os
import sys
from time import time

from datetime import datetime
from tensorboardX import SummaryWriter

from .average_meter import AverageMeter
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    def __init__(self, args, start_epoch, global_step):
        super(TrainLogger, self).__init__(args, start_epoch, global_step)

        self.metric_logs = {}
        self.metric_logs['loss'] = []

        self.best_loss = sys.maxsize
        self.loss_meter = AverageMeter()

        self.notImprovedCounter = 0


    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()
        self.loss_meter.reset()


    def log_iter(self, batch_loss):
        """Log results from a training iteration"""
        if self.iter % self.iters_per_print == 0:

            avg_time = time() - self.iter_start_time
            message = '[epoch: {}, iter: {}, time: {:.2f}, loss: {:.3g}]' \
                .format(self.epoch, self.iter, avg_time, 
                        batch_loss)

            self.write(message)

        self._log_scalars({'train-loss': batch_loss}, False)

        self.loss_meter.update(batch_loss)


    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += 1
        self.global_step += 1


    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))


    def end_epoch(self, metrics, optimizer):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            optimizer: Optimizer for the model.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}, average epoch loss: {:.3g}, best epoch loss: {:.3g}, not Improved for {} epochs, lr: {}]'
                   .format(self.epoch, time() - self.epoch_start_time, 
                           self.loss_meter.avg, 
                           self.best_loss, 
                           self.notImprovedCounter, 
                           optimizer.param_groups[0]['lr']))
        if metrics is not None:
            self._log_scalars(metrics)

        self.epoch += 1

    
    def has_improved(self):
        """Reports whether this epochs loss has improved since the last"""
        last_epoch_loss = self.loss_meter.avg
        isBetter = last_epoch_loss < self.best_loss
        if isBetter:
            self.best_loss = last_epoch_loss
            self.notImprovedCounter = 0
        else:
            self.notImprovedCounter += 1

        return isBetter

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
