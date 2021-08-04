'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512], interm_dim=128):
        
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0]) # h, w, c
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim) # 64->128
        self.FC2 = nn.Linear(num_channels[1], interm_dim) # 128->128
        self.FC3 = nn.Linear(num_channels[2], interm_dim) # 256->128
        self.FC4 = nn.Linear(num_channels[3], interm_dim) # 512->128

        self.linear = nn.Linear(4 * interm_dim, 1) # 4x128 = 512 -> 1
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))
        
        # concat 해야해서 interm_dim을 가짐
        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

# @author: Viet Nguyen <nhviet1009@gmail.com>
# """
# import torch
# import torch.nn as nn


# class Loss(nn.Module):
#     """
#         Implements the loss as the sum of the followings:
#         1. Confidence Loss: All labels, with hard negative mining
#         2. Localization Loss: Only on positive labels
#         Suppose input dboxes has the shape 8732x4
#     """

#     def __init__(self, dboxes):
#         super(Loss, self).__init__()
#         self.scale_xy = 1.0 / dboxes.scale_xy
#         self.scale_wh = 1.0 / dboxes.scale_wh

#         self.sl1_loss = nn.SmoothL1Loss(reduce=False)
#         self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
#         self.con_loss = nn.CrossEntropyLoss(reduce=False)

#     def loc_vec(self, loc):
#         gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
#         gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
#         return torch.cat((gxy, gwh), dim=1).contiguous()

#     def forward(self, ploc, plabel, gloc, glabel):
#         """
#             ploc, plabel: Nx4x8732, Nxlabel_numx8732
#                 predicted location and labels

#             gloc, glabel: Nx4x8732, Nx8732
#                 ground truth location and labels
#         """
#         mask = glabel > 0
#         pos_num = mask.sum(dim=1)

#         vec_gd = self.loc_vec(gloc)

#         # sum on four coordinates, and mask
#         sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
#         sl1 = (mask.float() * sl1).sum(dim=1)

#         # hard negative mining
#         con = self.con_loss(plabel, glabel)

#         # postive mask will never selected
#         con_neg = con.clone()
#         con_neg[mask] = 0
#         _, con_idx = con_neg.sort(dim=1, descending=True)
#         _, con_rank = con_idx.sort(dim=1)

#         # number of negative three times positive
#         neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
#         neg_mask = con_rank < neg_num

#         closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

#         # avoid no object detected
#         total_loss = sl1 + closs
#         num_mask = (pos_num > 0).float()
#         pos_num = pos_num.float().clamp(min=1e-6)
#         ret = (total_loss * num_mask / pos_num).mean(dim=0)
#         return ret

