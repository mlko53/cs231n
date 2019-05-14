import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from args import TrainArgParser
from dataloader import get_dataloader
from logger import TrainLogger
from loss import get_loss
from tqdm import tqdm
import models

import numpy as np
import random
import os


class PixelCNNSampler(object):
    def __init__(self, num_samples, batch_size, save_dir, device):
        self.batch_size = batch_size
        self.num_samples = num_samples 
        self.save_dir = save_dir
        self.device = device

    def sample(self, model, epoch):
        with torch.no_grad():
            model.eval()
            for i in range(self.num_samples):
                print("Sample{}".format(i))
                sample = torch.zeros(self.batch_size, 3, 64, 64).to(device)
                for x in tqdm(range(64)):
                    for y in range(64):
                        out = model(Variable(sample))
                        probs = F.softmax(out[:,:,x,y], dim=2).data
                        for c in range(3):
                            pixel = torch.multinomial(probs[:,c], 1).float() / 255.
                            sample[:,c,x,y] = pixel[:,0]

                image_path = os.path.join(self.save_dir, "epoch{}_sample{}.png".format(epoch, i))
                save_image(sample, image_path, nrow=3)
        
if __name__ == '__main__':
    parser = TrainArgParser()
    args = parser.parse_args()

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

    assert args.resume

    resume_path = os.path.join(args.save_dir, "best.pth.tar")
    print("Resuming from checkpoint at {}".format(resume_path))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

    print("Sampling...")
    sampler = PixelCNNSampler(5, args.batch_size, args.save_dir, device)
    sampler.sample(model, start_epoch)
