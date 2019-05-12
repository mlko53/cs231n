import torch.utils.data as data

from .randomDataset import RandomDataset


def get_dataloader(args, split):

    if args.dataset == "random":
        dataset = RandomDataset(split)
    else:
        raise ValueError("Dataset is not supported")

    loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=8)

    return loader
