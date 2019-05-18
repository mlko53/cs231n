import numpy as np
import os
import sys
from time import time
import torch

from datetime import datetime
from tensorboardX import SummaryWriter

from .average_meter import AverageMeter
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    def __init__(self, args, start_epoch, global_step):
        super(TrainLogger, self).__init__(args, start_epoch, global_step)

        self.metric_logs = {}
        self.metric_logs['loss'] = []

        self.val_best_loss = sys.maxsize
        self.loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()

        self.notImprovedCounter = 0


    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()


    def log_iter(self, batch_loss):
        """Log results from a training iteration"""
        if self.iter % self.iters_per_print == 0:

            avg_time = time() - self.iter_start_time
            message = '[epoch: {}, iter: {}, time: {:.2f}, loss: {:.5g}]' \
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
        self.loss_meter.reset()
        self.val_loss_meter.reset()


    def end_epoch(self, metrics, optimizer):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            optimizer: Optimizer for the model.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}, loss: {:.5g}, val-loss: {:.5g}, best val-loss: {:.5g}, not Improved for {} epochs, lr: {}]'
                   .format(self.epoch, time() - self.epoch_start_time, 
                           self.loss_meter.avg, 
                           self.val_loss_meter.avg,
                           self.val_best_loss, 
                           self.notImprovedCounter, 
                           optimizer.param_groups[0]['lr']))
        if metrics is not None:
            self._log_scalars(metrics)

        self.epoch += 1

    
    def has_improved(self, model, optimizer, j):
        """
        Reports whether this epochs loss has improved since the last
        Saves model if improvement
        """
        last_epoch_loss = self.val_loss_meter.avg
        isBetter = last_epoch_loss < self.val_best_loss
        self.val_best_loss = last_epoch_loss
        self.notImprovedCounter = 0
        # save model
        print("Saving...")
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': self.val_loss_meter.avg,
            'epoch': self.epoch,
            'iter': j
        }
        if isBetter:
            torch.save(state, os.path.join(self.save_dir, 'best.pth.tar'))
            torch.save(state, os.path.join(self.save_dir, 'current.pth.tar'))
        else:
            self.notImprovedCounter += 1
            torch.save(state, os.path.join(self.save_dir, 'current.pth.tar'))

        return isBetter

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
