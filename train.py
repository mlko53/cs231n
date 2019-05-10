import copy
import json
import numpy as np
import random
import torch
import torch.nn as nn

from args import TrainArgParser
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

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.resume:
        pass
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(args)
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(device)

if __name__ == '__main__':
    parser = TrainArgParser()
    main(parser.parse_args())
