"""
modules.py - This file stores the rathering boring network blocks.
"""

from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class MaskRGBEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f, m):

        f = torch.cat([f, m], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        return x


class MaskRGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet50(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f, m, o):

        f = torch.cat([f, m, o], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 256
        x = self.layer2(x) # 1/8, 512
        x = self.layer3(x) # 1/16, 1024

        return x
 

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet = models.resnet50(pretrained=True)
        resnet = mod_resnet.resnet50(pretrained=True)     #use mod_resnet as backbone
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv1 = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.skip_conv2 = ResBlock(up_c, up_c)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv2(self.skip_conv1(skip_f))
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_proj = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)
 
    def forward(self, x):  
        return self.key_proj(x), self.val_proj(x)

class Score(nn.Module):
    def __init__(self, input_chan):
        super(Score, self).__init__()
        input_channels = input_chan
        self.conv1 = nn.Conv2d(input_channels, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, 1)

        self.fc1 = nn.Linear(256*12*12, 1024)       #fc layers
        self.fc2 = nn.Linear(1024, 1)

        # self.gav = nn.AdaptiveAvgPool2d(1)        #global average pooling layers
        # self.fc = nn.Linear(256, 1)

        for i in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(i.bias, 0)

        for i in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(i.weight, a=1)
            nn.init.constant_(i.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = self.gav(x)               #the method of gav
        # x = x.view(x.size(0), -1)
        # x = F.leaky_relu(self.fc(x))
        return x

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, 
        padding=padding, dilation=dilation, bias=False)

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)      
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride=16, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                    BatchNorm(256), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 1024, 1, bias=False)
        self.bn1 = BatchNorm(1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
