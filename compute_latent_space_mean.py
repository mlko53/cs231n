import argparse
import numpy as np
import os
import random
import torch

from dataloader import *
from tqdm import tqdm
import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute GLOW latent space conditional mean")

    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--input_c', default=1, type=int, choices=[1,3])
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--size', default=128, type=int)

    args = parser.parse_args()

    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    print(device)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Build Model
    print("Building model...")
    model_fn = models.__dict__[args.model]
    model = model_fn(args, device)
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)
    
    # Load model with best val loss
    resume_path = os.path.join(args.save_dir, "best.pth.tar")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    start_iter = checkpoint['iter']
    print("Starting from epoch {}, iter {}".format(start_epoch, start_iter))

    print(MAIN_CATEGORIES)
