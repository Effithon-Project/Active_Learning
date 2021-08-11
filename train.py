"""
KITTI-dataset
"""
import os
import shutil
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, ResNet 
from src.utils import generate_dboxes, Encoder, kitti_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train#, evaluate
from src.dataset import collate_fn, KittiDataset

from torchvision.datasets import Kitti
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")

def main():

    torch.manual_seed(123)
    num_gpus = 1
    BATCH = 1

#     train_params = {"batch_size": BATCH * num_gpus,
#                     "shuffle": True,
#                     "drop_last": False,
#                     "num_workers": 1,
#                     "collate_fn": collate_fn}
    
    train_params = {"batch_size": 8, "collate_fn": collate_fn}

#     test_params = {"batch_size": opt.batch_size * num_gpus,
#                    "shuffle": False,
#                    "drop_last": False,
#                    "num_workers": opt.num_workers,
#                    "collate_fn": collate_fn}

    MODEL = "ssd"
    dboxes = generate_dboxes(model=MODEL)
    model = SSD(backbone=ResNet(), num_classes=len(kitti_classes))

    train_set =  KittiDataset("D:\\", train=True, transform=SSDTransformer(dboxes, (300, 300),val=False))

    train_loader = DataLoader(train_set , **train_params)
#     test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
#     test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)
    
    
    LR=2.6e-3

    LR = LR * num_gpus * (BATCH / 32)
    criterion = Loss(dboxes)
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)
    
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[43, 54], gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        
    first_epoch = 0
    EPOCHS = 1

    for epoch in range(first_epoch, EPOCHS):
        
        train(model, train_loader, epoch, criterion, optimizer, scheduler)
        nms_threshold = 0.5
        evaluate(model, test_loader, epoch, encoder, nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.module.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    
    # opt = get_args()
    main()
