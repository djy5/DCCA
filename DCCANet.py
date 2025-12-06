
import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from timm import create_model
from timm.models.layers import DropPath, trunc_normal_
from models.cd_modules import Decoder
from models.DCT import *
from models.CGILM import *
from models.DF import *


class mymodel(nn.Module):
    def __init__(self, embed_dim=256, encoder_type='resnet18', encoder_dims=[64, 128, 256, 512], freeze_backbone=False,
                 window_size=8):
        super().__init__()

        self.visual_encoder = create_model(encoder_type, pretrained=False, features_only=True)
        self.decoder = Decoder(64)

        self.apply(self._init_weights)
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.fca1 = MultiSpectralAttentionLayer(encoder_dims[0], c2wh[encoder_dims[0]], c2wh[encoder_dims[0]],
                                                reduction=16, freq_sel_method='top16')
        self.fca2 = MultiSpectralAttentionLayer(encoder_dims[1], c2wh[encoder_dims[1]], c2wh[encoder_dims[1]],
                                                reduction=16, freq_sel_method='top16')
        self.fca3 = MultiSpectralAttentionLayer(encoder_dims[2], c2wh[encoder_dims[2]], c2wh[encoder_dims[2]],
                                                reduction=16, freq_sel_method='top16')
        self.fca4 = MultiSpectralAttentionLayer(encoder_dims[3], c2wh[encoder_dims[3]], c2wh[encoder_dims[3]],
                                                reduction=16, freq_sel_method='top16')

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)

        self.num_images = 0

        self.df1 = DF(64, 64)
        self.df2 = DF(128, 128)
        self.df3 = DF(256, 256)
        self.df4 = DF(512, 512)

        self.cgil1 = CGIL(64, 8, 8)
        self.cgil2 = CGIL(128, 8, 8)
        self.cgil3 = CGIL(256, 8, 8)
        self.cgil4 = CGIL(512, 8, 8)


        self.conv_adjust_C4 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_adjust_C3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_adjust_C2 = nn.Conv2d(128, 64, kernel_size=1)


        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_visual_features(self, x):
        _, x1, x2, x3, x4 = self.visual_encoder(x)
        return x1, x2, x3, x4


    def forward(self, pre_img, post_img):
        outputs = []
        # extract visual features
        x1, x2, x3, x4 = self.forward_visual_features(pre_img)
        y1, y2, y3, y4 = self.forward_visual_features(post_img)

        x1_1 = self.fca1(x1)
        x2_1 = self.fca2(x2)
        x3_1 = self.fca3(x3)
        x4_1 = self.fca4(x4)

        y1_1 = self.fca1(y1)
        y2_1 = self.fca2(y2)
        y3_1 = self.fca3(y3)
        y4_1 = self.fca4(y4)



        F1 = self.df1(x1_1, y1_1)#[1, 64, 64, 64]
        F2 = self.df2(x2_1, y2_1)#[1, 128, 32, 32]
        F3 = self.df3(x3_1, y3_1)#[1, 256, 16, 16]
        F4 = self.df4(x4_1, y4_1)#[1, 512, 8, 8]


        C4 = self.cgil4(F4)  # torch.Size([1, 512, 8, 8])
        C4_1 = F.interpolate(C4, size=(16, 16), mode='bilinear', align_corners=False)  # Resize to [1,512,16,16]
        conv_C4_1 = self.conv_adjust_C4(C4_1)  # This is the new convolutional layer to match the channels
        CAT1 = torch.add(conv_C4_1, F3)  # Add along the channel dimension# torch.Size([1, 256, 16, 16])

        C3 = self.cgil3(CAT1)  # torch.Size([1, 256, 16, 16])
        C3_1 = F.interpolate(C3, size=(32, 32), mode='bilinear', align_corners=False)  # Resize to [1,256,32,32]
        conv_C3_1 = self.conv_adjust_C3(C3_1)  # This is the new convolutional layer to match the channels
        CAT2 = torch.add(conv_C3_1, F2)  # Add along the channel dimension

        C2 = self.cgil2(CAT2)  # torch.Size([1, 128, 32, 32])
        C2_1 = F.interpolate(C2, size=(64, 64), mode='bilinear', align_corners=False)  # Resize to [1,128,64,64]
        conv_C2_1 = self.conv_adjust_C2(C2_1)  # This is the new convolutional layer to match the channels 64
        CAT3 = torch.add(conv_C2_1, F1)  # Add along the channel dimension

        C1 = self.cgil1(CAT3)  # torch.Size([1, 64, 64, 64])

        pred = self.decoder(C1)  # torch.Size([1, 1, 256, 256])


        return pred
