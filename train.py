import copy
import json
import numpy as np
import random
import time
import torch
import torch.nn as nn

from args import TrainArgParser
from dataloader import get_dataloader
from tqdm import tqdm
import models


def write_args(args):
    save_dir = args.save_dir
    copy_args = copy.deepcopy(args)
    copy_args.save_dir = save_dir.as_posix()
    """Save args as a JSON file"""
    with (save_dir / 'args.json').open('w') as f:
        args_dict = vars(copy_args)
        print(args_dict)
        json.dump(args_dict, f, indent=4, sort_keys=True)
        f.write('/n')

    print("Saved Arguments")

def main(args):
    write_args(args)

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
    if args.resume:
        pass
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(args, device)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)

    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")

    start_epoch = 0
    for i in range(start_epoch, args.num_epochs):
        for image, label in tqdm(train_loader):
            image = image.to(device)
            output = model(image)
            print(output.shape)


if __name__ == '__main__':
    parser = TrainArgParser()
    main(parser.parse_args())
