import argparse
from pathlib import Path


class TrainArgParser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Generative models on CheXpert')

	self.parser.add_argument("--name", type=str, default='debugging')
	self.parser.add_argument("--save_dir", type=str, default='experiments/')

	# model config
        parser.add_argument('--model', default='pixelCNN', type=str, choices=['pixelCNN', 'glow']
	parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
	parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
	parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')

        # train config
	parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
        parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
        parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
        parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
        parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
        parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

        # device config
        parser.add_argument('--benchmark', type=bool, default=True, help='Turn on CUDNN benchmarking')
        parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
