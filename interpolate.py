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
    parser.add_argument('--size', default=128, type=int)
    
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--nrow', default=8, type=int)

    args = parser.parse_args()
    save_dir = Path("./experiments") / "Glow" / args.name
    (save_dir / "latents").mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir

    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    print(device)

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

    print("Loading normal images")
    normal_data = ChexpertDataset("train", args.batch_size, args.size, args.input_c, NO_FINDING)

    print("Loading latent means and computing manipulation vector")
    no_finding = torch.load(args.save_dir / f"latents/{NO_FINDING}.pt")
    latents = {}
    for category in MAIN_CATEGORIES:
        path = args.save_dir / "latents/{}.pt".format(category)
        latents[category] = torch.load(path) - no_finding
        latents[category].to(device)
    
    print("Generating manipulation images")
    manipulations = list(latents.values())
    manipulations = torch.stack(manipulations)
    manipulations = manipulations.expand(len(MAIN_CATEGORIES), args.nrow,
                                         args.size, args.size)
    manipulations = manipulations[:,:,None,:,:]
    step = torch.Tensor(np.linspace(0., 2., args.nrow)).to(device)
    manipulations = torch.einsum('abcde,b->abcde', (manipulations, step))
    del latents
    for i in tqdm(range(args.num_samples)):
        with torch.no_grad():
            rand_idx = random.randint(0, len(normal_data))
            image = normal_data[rand_idx][None,:,:,:]
            image.to(device)
            save_image(image, args.save_dir / "latents/original_{}.png".format(i))
            z = model(image)[0]
            z = z[0]
            z = z.expand(len(MAIN_CATEGORIES), args.nrow, *z.size())
            z = z + manipulations
            for j in range(z.shape[0]):
                image = model(z[j], reverse=True)[0]
                z[j] = image
            z = z.view(-1, args.input_c, args.size, args.size)
            z = torch.sigmoid(z)
            save_image(z, args.save_dir / "latents/interpolation_{}.png".format(i), nrow=args.nrow)
