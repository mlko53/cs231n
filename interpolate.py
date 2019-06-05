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
    parser.add_argument('--nrow', default=7, type=int)

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

    def generate_from_latents(latents, alpha, args, name, dataset, gen_model):
        manipulations = list(latents.values())
        manipulations = torch.stack(manipulations)
        manipulations = manipulations.expand(len(latents), args.nrow,
                                             args.size, args.size)
        manipulations = manipulations[:,:,None,:,:]
        step = torch.Tensor(np.linspace(-alpha, alpha, args.nrow)).to(device)
        manipulations = torch.einsum('abcde,b->abcde', (manipulations, step))
        for i in tqdm(range(args.num_samples)):
            with torch.no_grad():
                rand_idx = random.randint(0, len(dataset))
                image = dataset[rand_idx][None,:,:,:]
                image.to(device)
                z = gen_model(image)[0]
                z = z[0]
                z = z.expand(len(latents), args.nrow, *z.size())
                z = z + manipulations
                for j in range(z.shape[0]):
                    image = gen_model.forward(z[j], True)[0]
                    z[j] = image
                z = z.view(-1, args.input_c, args.size, args.size)
                z = torch.sigmoid(z)
                save_image(z, args.save_dir / "latents/{}_{}.png".format(name, i), nrow=args.nrow)

    print("Generating manipulation images for age and sex")
    no_finding = torch.load(args.save_dir / f"latents/{NO_FINDING}.pt")
    latents = {}
    latents['80'] = torch.load(args.save_dir / "latents/8.pt") - no_finding
    generate_from_latents(latents, 1., args, "age_interpolation", normal_data, model)

    latents = {}
    latents['female'] = torch.load(args.save_dir / "latents/Female.pt") - torch.load(args.save_dir / "latents/Male.pt")
    generate_from_latents(latents, 1., args, "gender_interpolation", normal_data, model)

    """
    print("Generating manipulation images from main categories")
    no_finding = torch.load(args.save_dir / f"latents/{NO_FINDING}.pt")
    latents = {}
    for category in MAIN_CATEGORIES:
        path = args.save_dir / "latents/{}.pt".format(category)
        latents[category] = torch.load(path) - no_finding
        latents[category].to(device)
    generate_from_latents(latents, 1.5, args, "main_interpolation", normal_data, model)
    
    print("Generating manipulationg images for other categories")
    other_categories = list(set(ALL_CATEGORIES) - set(MAIN_CATEGORIES))
    latents = {}
    for category in other_categories:
        path = args.save_dir / "latents/{}.pt".format(category)
        latents[category] = torch.load(path) - no_finding
        latents[category].to(device)
    generate_from_latents(latents, 1.5, args, "other_interpolation", normal_data, model)

    print("Linear interpolation")
    del normal_data
    dataset = ChexpertDataset("train", args.batch_size, args.size, args.input_c, None)
    for i in range(args.num_samples):
        with torch.no_grad():
            x = random.randint(0, len(dataset))
            y = random.randint(0, len(dataset))
            image_x = dataset[x][None,:,:,:]
            image_y = dataset[y][None,:,:,:]
            z_x = model(image_x)[0][0]
            z_y = model(image_y)[0][0]
            z = []
            for a in np.linspace(0,1,9):
                mix = a * z_x + (1.-a)*z_y
                z.append(mix)
            z = torch.stack(z)
            image = model.forward(z, True)[0]
            image = torch.sigmoid(image)
            save_image(image, args.save_dir / "latents/linear_interpolation_{}.png".format(i), nrow=9)
      """
