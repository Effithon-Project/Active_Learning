import numpy as np
import itertools
from math import sqrt

import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou, box_convert


kitti_classes = ['Car','DontCare','Person_sitting','Truck','Cyclist','Pedestrian','Misc','Van','Tram']

def generate_dboxes(model="ssd"):
        
    figsize = (384, 1280)

    feat_size =  [(48, 160), (24, 80), (12, 40), (6, 20), (4, 18), (2, 16)]

    steps = [8, 16, 32, 64, 100, 300]

    scales = [(21., 45.),(45., 99.),(99., 153.),(153., 207.),(207., 261.),(261., 315.), (315,345)]

    aspect_ratios = [[2, .5],
                    [2, .5, 3, 1./3],
                    [2, .5, 3, 1./3],
                    [2, .5, 3, 1./3],
                    [2, .5],
                    [2, .5]]

    #  DefaultBoxes------------------------------------------------------->
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)

    return dboxes

class DefaultBoxes(object):
    
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        
        self.fig_size = fig_size                  
        self.feat_size = feat_size                
        self.steps = steps                        
        self.scales = scales                      
        self.aspect_ratios = aspect_ratios        
        self.scale_xy = scale_xy                  
        self.scale_wh = scale_wh                  
        
        fk_w = fig_size[1] / np.array(steps)
        fk_h = fig_size[0] / np.array(steps)

        
        self.default_boxes = []
        
        for idx, sfeat in enumerate(self.feat_size):

            sk1_h = scales[idx][0] / fig_size[0] # 300
            sk1_w = scales[idx][1] / fig_size[1] # 1000
#             print(idx)
            sk2_h = scales[idx + 1][0] / fig_size[0]
            sk2_w = scales[idx + 1][1] / fig_size[1]

            sk3_h = sqrt(sk1_h * sk2_h)
            sk3_w = sqrt(sk1_w * sk2_w)
            
            all_sizes = [(sk1_w, sk1_h), (sk3_w, sk3_h)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1_w * sqrt(alpha), sk1_h / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
                
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat[0]), range(sfeat[1])):

                    cx, cy = (j + 0.5) / fk_w[idx], (i + 0.5) / fk_h[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        
        self.dboxes_ltrb = box_convert(self.dboxes, in_fmt="cxcywh", out_fmt="xyxy")

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        else:  # order == "xywh"
            return self.dboxes


class Encoder(object):


    def __init__(self, dboxes):
        
        self.dboxes = dboxes(order="ltrb") # left top right bottom
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
#         print(self.dboxes_xywh.size())
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):

        ious = box_iou(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
        bboxes_in = box_convert(bboxes_in, in_fmt="cxcywh", out_fmt="xyxy")

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, nms_threshold=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_threshold, max_output))
        return output

    def decode_single(self, bboxes_in, scores_in, nms_threshold, max_output, max_num=200):
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < nms_threshold
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_threshold]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
                                             torch.tensor(labels_out, dtype=torch.long), \
                                             torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]




