'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import os
import random
import numpy as np
# import visdom
from tqdm import tqdm
import shutil

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# nn
import torch
import torch.nn as nn

# learning
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR
# torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import Kitti 

# custom utils from base code
from config import *
import src.lossnet as lossnet
from src.sampler import SubsetSequentialSampler 
from src.transform import SSDTransformer
from src.model import SSD, ResNet
from src.utils import generate_dboxes, Encoder, kitti_classes
from src.loss import Loss
from src.dataset import collate_fn, KittiDataset
# map
from src.metric import *

import warnings
warnings.filterwarnings("ignore")


# seed
random.seed("Jungyeon")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True # reproduction을 위한 부분
# https://pytorch.org/docs/stable/notes/randomness.html

dboxes = generate_dboxes(model="ssd")
encoder = Encoder(dboxes)
# directory you download 'D:\\'
kitti_tot = KittiDataset("D:\\", train=True,
                         transform=SSDTransformer(dboxes, (300, 300),val=False))
        
kitti_unlabeled = KittiDataset("D:\\", train=True,
                               transform=SSDTransformer(dboxes, (300, 300),val=False))

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    base paper's core contribution
    """
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] 

    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 
    # 1 operation which is defined by the authors
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


iters = 0

def train_epoch(models,
                dataloaders,
                epoch,
                epoch_loss,
                criterion,
                optimizers,
                schedulers,
                vis=None,
                plot_data=None):
    
    models['backbone'].train()
    models['module'].train()
    train_loader = dataloaders['train']
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)


    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        img = img.cuda()
        gloc = gloc.cuda() # gt localization
        glabel = glabel.cuda() # gt label
        
        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        # locs, confs // predicted localization, predicted label
        ploc, plabel, out_dict = models['backbone'](img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        target_loss = criterion(ploc, plabel, gloc, glabel) # confidence 기반
        
        features = out_dict
        pred_loss = models['module'](features) 
        pred_loss = pred_loss.view(pred_loss.size(0))
        
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        
        #----------------------LossPredLoss---------------------------
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss
        
        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))
        
        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()
            

def test(models, dataloaders, encoder, nms_threshold, mode='val'):
    """
    evaluate(model, test_loader, encoder, nms_threshold)
    """
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()
    
    detections = []
    category_ids = [i for i in range(9)]
    test_loader = dataloaders['test']
    
    for nbatch, (img, img_id, img_size, gloc, glabel) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")

        img = img.cuda()
        gloc = gloc.cuda() # gt localization
        glabel = glabel.cuda() # gt label

        with torch.no_grad():
            # Get predictions
            ploc, plabel, out_dict = models['backbone'](img)
            ploc, plabel = ploc.float(), plabel.float()
#             print(ploc)
            gloc = gloc.transpose(1, 2).contiguous()
            
            # batch 묶음에서 이미지 하나 가져오기 idx:0,1,2,3,4,...
            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i,
                                                  plabel_i,
                                                  nms_threshold,
                                                  200)[0]
#                     print(result[0].size()) # torch.Size([200, 4]) bbox
                except:
                    print("No object detected in idx: {}".format(idx))

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result] # numpy
                for loc_, label_, prob_ in zip(loc, label, prob):
                    # xyxy
                    print(loc_[0] * width,
                          loc_[1] * height,
                          loc_[2] * width,
                          loc_[3] * height)
                
                    detections.append([img_id[idx],
                                       loc_[0] * width,
                                       loc_[1] * height,
                                       loc_[2] * width,
                                       loc_[3] * height,
                                       prob_,
                                       category_ids[label_ - 1]])

    print()
    print(len(detections))
#     detections = np.array(detections, dtype=np.float32)
#     return len(detections)
    


def train(models,
          criterion,
          optimizers,
          schedulers,
          dataloaders,
          num_epochs,
          epoch_loss,
          vis=None,
          plot_data=None):

    print('>> Train a Model.')
    
    best_acc = 0.
    checkpoint_dir = os.path.join('./ckpt', 'train', 'weights')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        
        schedulers['backbone'].step()
        schedulers['module'].step()
        
        # -----------------EPOCH--------------------------
        train_epoch(models,
                    dataloaders,
                    epoch,
                    epoch_loss,
                    criterion,
                    optimizers,
                    schedulers,
                    vis,
                    plot_data)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            pass
#             acc = test(models, dataloaders, encoder, nms_threshold, mode='test')
#             if best_acc < acc:
#                 best_acc = acc
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'state_dict_backbone': models['backbone'].state_dict(),
#                     'state_dict_module': models['module'].state_dict()},
#                     '%s/active_ssd_kitti.pth' % (checkpoint_dir))
                
#             print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
            
    print('>> Finished.')
    
    
def get_uncertainty(models, unlabeled_loader):
    """
    data selecting을 위한 uncertainty 계산
    """
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for i, (img, _, _, gloc, glabel) in enumerate(unlabeled_loader):
            img = img.cuda()
#             gloc = gloc.cuda() # gt localization
#             glabel = glabel.cuda() # gt label
            
            ploc, plabel, out_dict = models['backbone'](img)

            features = out_dict
            pred_loss = models['module'](features) 
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


if __name__ == '__main__':

#     vis = visdom.Visdom(server='http://localhost', port=9000)
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(TRIALS):
        # train : test = 7:3
        tot_indices = list(range(NUM_TOT)) 
        random.shuffle(tot_indices)
        test_set = tot_indices[:NUM_TEST]
        train_set = tot_indices[NUM_TEST:]
        
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=300
        random.shuffle(train_set)
        
        labeled_set = train_set[:ADDENDUM]
        unlabeled_set = train_set[ADDENDUM:]  
        
        train_params = {"batch_size": BATCH,
                        "shuffle": False, # sampler랑 같이 쓸 수 없음
                        "drop_last": False,
                        "collate_fn": collate_fn,
                        "sampler": SubsetRandomSampler(labeled_set),
                        "pin_memory":True}

        test_params = {"batch_size": BATCH,
                       "shuffle": False,
                       "drop_last": False,
                       "collate_fn": collate_fn,
                       "sampler": SubsetRandomSampler(test_set)}

        train_loader = DataLoader(kitti_tot, **train_params)
        test_loader = DataLoader(kitti_tot, **test_params)
        
        dataloaders  = {'train': train_loader, 'test': test_loader}        
        
        # backbone
        model = SSD(backbone=ResNet(), num_classes=len(kitti_classes)).cuda()
        # Loss model
        loss_module = lossnet.LossNet().cuda() 
        
        models      = {'backbone': model, 'module': loss_module}
        
        torch.backends.cudnn.benchmark = False
        
        # Active learning cycles 
        for cycle in range(CYCLES):
            LR = LR * (BATCH / 32)
            criterion = Loss(dboxes).cuda()

            MOMENTUM = 0.9
            WEIGHT_DECAY = 0.0005
        
            optim_backbone = torch.optim.SGD(models['backbone'].parameters(),
                                             lr=LR,
                                             momentum=MOMENTUM,
                                             weight_decay=WDECAY,
                                             nesterov=True)
            
            optim_module = torch.optim.SGD(models['module'].parameters(),
                                           lr=LR,
                                           momentum=MOMENTUM,
                                           weight_decay=WDECAY,
                                           nesterov=True)
            
            sched_backbone = MultiStepLR(optimizer=optim_backbone,
                                         milestones=MILESTONES,
                                         gamma=0.1)
            
            sched_module = MultiStepLR(optimizer=optim_module,
                                       milestones=MILESTONES,
                                       gamma=0.1)
            
            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}
                
            ##----------------------TRAIN----------------------------
            train(models,
                  criterion,
                  optimizers,
                  schedulers,
                  dataloaders,
                  EPOCH,
                  EPOCHL)
#                   vis,
#                   plot_data)
            
            nms_threshold = 0.5
            mAP = test(models, dataloaders, encoder, nms_threshold, mode='test')

            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test detect_num {}'.format(trial+1,
                                                                                        TRIALS,
                                                                                        cycle+1,
                                                                                        CYCLES,
                                                                                        len(labeled_set),
                                                                                        detect_num))

            # Update the labeled dataset via loss prediction-based uncertainty measurement
            # Randomly sample 300 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            # more convenient if we maintain the "order" of subset
            unlabeled_params = {"batch_size": BATCH,
                                "collate_fn": collate_fn,
                                "sampler": SubsetSequentialSampler(subset),
                                "pin_memory":True}
            
            unlabeled_loader = DataLoader(kitti_unlabeled, **unlabeled_params)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)
            # 나중에 강화학습을 넣을 수 있지 않을까(휴리스틱하거나 mathmatics 적인 부분이니까)
            
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            train_params = {"batch_size": BATCH,
                            "shuffle": False, # sampler랑 같이 쓸 수 없음
                            "drop_last": False,
                            "collate_fn": collate_fn,
                            "sampler": SubsetRandomSampler(labeled_set),
                            "pin_memory":True}
            
            dataloaders['train'] = DataLoader(kitti_tot, **train_params)
            print('>> Datasets are Updated.')
            print('>> LABELED:', len(labeled_set))
            print('>> UNLABELED', len(unlabeled_set))
        # Save a checkpoint
        torch.save({'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()},
            './ckpt/train/weights/active_resnet50_kitti_trial{}.pth'.format(trial))