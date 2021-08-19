'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F


class LossNet(nn.Module):
    """
    feature_sizes바꿀려면 dboxes_generate 바꿔야함
    num_channels <- model.py out_channels
    """
    def __init__(self, feature_sizes=[32, 16, 8, 4, 2],
                 num_channels=[1024, 512, 512, 256, 256, 256], interm_dim=128):
        
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0]) # h, w, c
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])
        self.GAP5 = nn.AvgPool2d(feature_sizes[4])

        self.FC1 = nn.Linear(num_channels[0], interm_dim) # 16->128
        self.FC2 = nn.Linear(num_channels[1], interm_dim) # 32->128
        self.FC3 = nn.Linear(num_channels[2], interm_dim) # 64->128
        self.FC4 = nn.Linear(num_channels[3], interm_dim) # 128->128
        self.FC5 = nn.Linear(num_channels[4], interm_dim) # 256->128


        self.linear = nn.Linear(5 * interm_dim, 1) # 5x128 = 640 -> 1
    
    def forward(self, features):
        """
        torch.Size([4, 1024, 38, 38])
        torch.Size([4, 512, 19, 19])
        torch.Size([4, 512, 10, 10])
        torch.Size([4, 256, 5, 5])
        torch.Size([4, 256, 3, 3])
        """
#         print(features[0].size())
#         print(features[1].size())
#         print(features[2].size())
#         print(features[3].size())
#         print(features[4].size())
        out1 = self.GAP1(features[0]) # 32
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1]) # 16
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2]) # 8
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3]) # 4
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))
        
        out5 = self.GAP5(features[4]) # 2
        out5 = out5.view(out5.size(0), -1)
        out5 = F.relu(self.FC5(out5))
        
        # concat 해야해서 interm_dim을 가짐
        out = self.linear(torch.cat((out1, out2, out3, out4, out5), 1))
        return out


