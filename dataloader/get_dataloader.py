import torch.utils.data as data

from .chexpertDataset import ChexpertDataset
from .randomDataset import RandomDataset


def get_dataloader(args, split, pathology=None):

    if args.dataset == "random":
        dataset = RandomDataset(split, args.batch_size)
    elif args.dataset == "chexpert":
        dataset = ChexpertDataset(split, args.batch_size, args.size, args.input_c, pathology)
    else:
        raise ValueError("Dataset is not supported")

    loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=8)

    return loader
