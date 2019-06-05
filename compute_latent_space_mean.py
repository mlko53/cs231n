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

    print("Computing mean from these pathology")
    print(ALL_CATEGORIES)
    for category in ALL_CATEGORIES:
        print("[{}]".format(category))
        dataloader = get_dataloader(args, "train", category)
        running_avg = torch.zeros((args.input_c, args.size, args.size)).to(device)
        with torch.no_grad():
            for i, image in tqdm(enumerate(dataloader)):
                image = image.to(device)
                z = model(image)[0]
                z = z.mean(dim=0)
                running_avg = (i / (i+1)) * running_avg + (1 / (i+1)) * z
            torch.save(running_avg, args.save_dir / "latents/{}.pt".format(category))
            sample, _ = model(torch.stack([running_avg]*4), reverse=True) # make batch size 4 so that model doesn't complain
            sample = torch.sigmoid(sample)
            save_image(sample[0,:,:,:], args.save_dir / "latents/{}.png".format(category))
