'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
# ?? 
# --

import os
import random
import numpy as np
import visdom
from tqdm import tqdm

#data loader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler #??
# nn
import torch
import torch.nn as nn
# learning
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import Kitti 

# custom utils from base code
import models.resnet as resnet
import models.lossnet2 as lossnet2
from config import *
from data.sampler import SubsetSequentialSampler #?? 아마도 라벨을 추가하는 과정에서 필요한 것 같음

# seed
random.seed("Jungyeon")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True # reproduction을 위한 부분
# https://pytorch.org/docs/stable/notes/randomness.html

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) 
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # for CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) 
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # for CIFAR-100
])

# data load and transform

# directory you download 'D:\\'
kitti_train = Kitti('D:\\', train=True, download=True)
kitti_unlabeled = Kitti('D:\\', train=True, download=True)
kitti_test  = Kitti('D:\\', train=False, download=True)

#------------------------------Loss Prediction Loss------------------------------
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



#------------------------------Training-------------------------------
iters = 0

def train_epoch(models, criterion, optimizers,
                dataloaders, epoch, epoch_loss,
                vis=None, plot_data=None):
    
    models['backbone'].train()
    models['module'].train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            # heuristic
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
            
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

        #--------------------------------Visualize--------------------------------
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                m_backbone_loss.item(),
                m_module_loss.item(),
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )
            

def test(models, dataloaders, mode='val'):
    """
    모델을 학습시킨 후  loss 구하기 위해 
    """
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

def train(models, criterion, optimizers,
          schedulers, dataloaders, num_epochs,
          epoch_loss, vis, plot_data):
    """
    진짜 모델 훈련
    """
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()
        # 위에서 정의한 함수
        train_epoch(models, criterion,
                    optimizers, dataloaders,
                    epoch, epoch_loss, vis, plot_data)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                '%s/active_resnet50_kitti.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
    
    
def get_uncertainty(models, unlabeled_loader):
    """
    data selecting을 위한 uncertainty 계산
    """
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) 
            # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu() #??


if __name__ == '__main__':
    # 시각화
    vis = visdom.Visdom(server='http://localhost', port=9000)
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN)) # 데이터 전체 갯수 cifar는 50000만장
        random.shuffle(indices)
        # 라벨링 유무
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        # 서브샘플러로 라벨 데이터를 가져온다
        train_loader = DataLoader(kitti_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        
        test_loader  = DataLoader(kitti_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        # Target model
        resnet50    = resnet.ResNet50(num_classes=9).cuda() #kitti는 9개 클래스로 오브젝트 디텍션 진행
        # Loss model
        loss_module = lossnet2.LossNet().cuda() # lossnet을 위한 feature 빼오는 부분
        
        models      = {'backbone': resnet50, 'module': loss_module}
        
        torch.backends.cudnn.benchmark = False

        # Active learning cycles # config에서 10번으로 잡음 - 1000x10 = 10000개까지 데이터 라벨이 들어감
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            # object detection일 경우 loss 식 바꿔야함
            
            
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
            
            acc = test(models, dataloaders, mode='test')
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1,
                                                                                        TRIALS,
                                                                                        cycle+1,
                                                                                        CYCLES,
                                                                                        len(labeled_set),
                                                                                        acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement
            # 모델에 추가적으로 라벨을 넣어줄 부분
            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the "order" of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            # 이 부분이 data selection이 일어나는 부분
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)
            # 이 부분에 나중에 강화학습을 넣을 수 있지 않을까 기대(왜냐하면 휴리스틱하거나 mathmatics 적인 부분이니까)
            
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
        
        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                './cifar10/train/weights/active_resnet50_kitti_trial{}.pth'.format(trial))