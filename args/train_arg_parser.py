import argparse
from pathlib import Path


class TrainArgParser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Generative models on CheXpert')
        
        self.parser.add_argument("--name", type=str, default='debugging')
        self.parser.add_argument("--save_dir", type=str, default='experiments/')

        # dataset config
        self.parser.add_argument("--dataset", type=str, default='random')
        self.parser.add_argument("--split", type=str, default='train', choices=['train', 'val', 'test'])

	# model config
        self.parser.add_argument('--model', default='PixelCNN', type=str, choices=['PixelCNN', 'Glow'])
        self.parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
        self.parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
        self.parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')

        # train config
        self.parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
        self.parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
        self.parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
        self.parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
        self.parser.add_argument('--resume', type=bool, default=False, help='Resume from checkpoint')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

        # device config
        self.parser.add_argument('--benchmark', type=bool, default=True, help='Turn on CUDNN benchmarking')
        self.parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')


    def parse_args(self):
        args = self.parser.parse_args()

        # create path
        save_dir = (Path(args.save_dir) / 
                    args.model / 
                    args.name) 
        save_dir.mkdir(parents=True, exist_ok=True)
        args.save_dir = save_dir
        return args
