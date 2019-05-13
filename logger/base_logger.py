import os

from datetime import datetime
from tensorboardX import SummaryWriter


class BaseLogger(object):
    def __init__(self, args, start_epoch, global_step):
        
        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H%M'))
        self.summary_writer = SummaryWriter(log_dir = log_dir)

        self.iters_per_print = args.iters_per_print
        self.epoch = start_epoch
        self.iter = 0
        self.global_step = global_step
        self.iter_start_time = None
        self.epoch_start_time = None

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard"""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')
            self.summary_writer.add_scalar(k, v, self.global_step)

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration"""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration"""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, optimizer):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError


