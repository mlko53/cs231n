import copy
import json
import numpy as np
import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

from args import TrainArgParser
from dataloader import get_dataloader
from logger import TrainLogger
from loss import get_loss
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
    model_fn = models.__dict__[args.model]
    model = model_fn(args, device)
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)

    # Loss fn
    loss_fn = get_loss(args.model)

    # Data loaders
    train_loader = get_dataloader(args, "train")
    val_loader = get_dataloader(args, "val")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logger and Resume
    if args.resume:
        resume_path = os.path.join(args.save_dir, "best.pth.tar")
        print("Resuming from checkpoint at {}".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(train_loader)
        logger = TrainLogger(args, start_epoch, global_step)
        logger.best_val_loss = checkpoint['val_loss']
    else:
        start_epoch = 0
        global_step = 0
        logger = TrainLogger(args, start_epoch, global_step)

    for i in range(start_epoch, args.num_epochs):

        # Train
        model.train()
        logger.start_epoch()
        for image in train_loader:
            logger.start_iter()

            image = image.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, image)
            loss.backward()
            optimizer.step()

            logger.log_iter(loss)
            logger.end_iter()

        # Eval
        model.eval()
        for image in tqdm(val_loader):
            image = image.to(device)
            loss = loss_fn(output, image)
            logger.val_loss_meter.update(loss)

        logger.has_improved(model)
        logger.end_epoch({'val-loss': logger.val_loss_meter.avg}, optimizer)


if __name__ == '__main__':
    parser = TrainArgParser()
    main(parser.parse_args())
