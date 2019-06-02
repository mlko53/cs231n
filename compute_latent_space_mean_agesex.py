import argparse
import numpy as np
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
from torchvision.utils import save_image

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
    save_dir = Path("./experiments") / "Glow" / args.name
    (save_dir / "latents").mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir

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

    # Set model to eval
    model.eval()

    latents = {}
    latents["Female"] = (torch.zeros((args.input_c, args.size, args.size)).to(device), 0)
    latents["Male"] = (torch.zeros((args.input_c, args.size, args.size)).to(device), 0)
    for i in range(10):
        latents[str(i)] = (torch.zeros((args.input_c, args.size, args.size)).to(device), 0)

    dataloader = ChexpertDataset("train", args.batch_size, args.size, args.input_c, None, agesex=True)
    for i in tqdm(range(len(dataloader))):
        image, sex, age = dataloader[i]
        image = image[None,:,:,:]
        image.to(device)
        z = model(image)[0]
        z = z[0]

        # update sex latents
        c = latents[sex][1]
        m = latents[sex][0]
        latents[sex][0] = (c / (c+1))*m + (1/(c+1))*z
        latents[sex][1] = c+1

        # update age latents
        c = lantents[str(age // 10)][1]
        m = latents[str(age // 10)][0]
