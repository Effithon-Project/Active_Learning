import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
#             print("-"*100)
#             print("ploc from one feature map: ",(l(s).view(s.size(0), 4, -1)).size())
#             print("plabel from one feature map: ",(c(s).view(s.size(0),self.num_classes, -1)).size())
            ret.append((l(s).view(s.size(0), 4, -1),
                        c(s).view(s.size(0),self.num_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        locs = torch.clamp(locs, min=0, max=1) ##--------------------------------------------
        confs = torch.clamp(confs, min=0)
#         print(locs)
#         print ((locs <= 0 ).nonzero(as_tuple=True)[0])

        return locs, confs

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(Base):
    def __init__(self, backbone=ResNet(), num_classes=9):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes

        self._build_additional_features(self.feature_extractor.out_channels) # oc [1024, 512, 512, 256, 256, 256]
        self.num_defaults = [4, 6, 6, 6, 4, 4] # nd
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])): ##----------
            if i < 4:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)


    def forward(self, x):
        
        x = self.feature_extractor(x)
        detection_feed = [x]
        
        # out 1,2,3,4,5
        out_dict = {}
        for i, l in enumerate(self.additional_blocks):
            out_dict[i] = x
            x = l(x)
            detection_feed.append(x)
        
#         print(detection_feed[0].size())
#         print(detection_feed[1].size())
#         print(detection_feed[2].size())
#         print(detection_feed[3].size())
#         print(detection_feed[4].size())
#         print(detection_feed[5].size())
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
#         print(locs.size())
        return locs, confs, out_dict