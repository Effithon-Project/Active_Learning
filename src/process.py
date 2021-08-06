"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
# from pycocotools.cocoeval import COCOeval
# from apex import amp

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

def train(models, criterion, optimizers,
          schedulers,
          dataloaders,
          epoch,
          epoch_loss,
          vis,
          plot_data): # , is_amp
    """
    통합 중 
    - vis는 1
    - 로스 체계 1
    - 로스 모듈은 2
    """
    print('>> Train a Model.')
    
    best_acc = 0.
    checkpoint_dir = os.path.join('./kitti', 'train', 'weights')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()
    
        # train_epoch
        train_epoch(models, criterion,
                    optimizers, dataloaders,
                    epoch, epoch_loss, vis, plot_data)
        
    models['backbone'].train()
    num_iter_per_epoch = len(dataloaders["train"])
    progress_bar = tqdm(ataloaders["train"])
    scheduler.step()
    
    
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        # (img, _, _, gloc, glabel) == data
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
# def train(model, train_loader, epoch, writer, criterion, optimizer, scheduler): # , is_amp
#     """
#     통합 중 
#     - vis는 1
#     - 로스 체계 1
#     - 로스 모듈은 2
#     """
    
#     model.train()
#     num_iter_per_epoch = len(train_loader)
#     progress_bar = tqdm(train_loader)
#     scheduler.step()
#     for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
#         if torch.cuda.is_available():
#             img = img.cuda()
#             gloc = gloc.cuda()
#             glabel = glabel.cuda()

#         ploc, plabel = model(img)
#         ploc, plabel = ploc.float(), plabel.float()
#         gloc = gloc.transpose(1, 2).contiguous()
#         loss = criterion(ploc, plabel, gloc, glabel)

#         progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

#         writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

# #         if is_amp:
# #             with amp.scale_loss(loss, optimizer) as scale_loss:
# #                 scale_loss.backward()
# #         else:
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()


def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx],
                                       loc_[0] * width,
                                       loc_[1] * height,
                                       (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height,
                                       prob_,
                                       category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)

    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
