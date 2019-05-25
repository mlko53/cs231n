import argparse
import numpy as np
import os
from pathlib import Path
import random
import torch
import torch.nn as nn

from dataloader import *
from tqdm import tqdm
import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute GLOW latent space conditional mean")

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dataset', default='chexpert', type=str)
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--input_c', default=1, type=int, choices=[1,3])
    parser.add_argument('--model', default="Glow", type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--size', default=128, type=int)

    args = parser.parse_args()
    args.save_dir = Path("./experiments") / "Glow" / args.name

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

    print("Computing mean from these pathology")
    print(MAIN_CATEGORIES)
    for category in MAIN_CATEGORIES:
        print("[{}]".format(category))
        dataloader = get_dataloader(args, "train", category)
