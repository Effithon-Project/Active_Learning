"""
KITTI-dataset
"""
import numpy as np
from tqdm.autonotebook import tqdm
import torch
# from pycocotools.cocoeval import COCOeval


def train(model, train_loader, epoch, criterion, optimizer, scheduler):
    
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    scheduler.step()
#     print("here!!!!!!!")
#     print(progress_bar)
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()
        
        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)
#         if i==10:
#             print("-"*100)
#             print("img: ", img[0])
#             print(img[0].size())
#             print("-"*100)
#             print("gloc: ", gloc[0])
#             print(gloc[0].size())
#             print("-"*100)
#             print("glabel: ", glabel[0])
#             print(glabel[0].size())
#             print("-"*100)
#             print("ploc: ", ploc[0])
#             print(ploc[0].size())
#             print("-"*100)
#             print("plabel: ", plabel[0])
#             print(plabel[0].size())
#             print("-"*100)
#             print("loss: ", loss)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, test_loader, epoch, encoder, nms_threshold):
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
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)
    print(detections)

#     coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
