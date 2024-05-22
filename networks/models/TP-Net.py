import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import Res3D_block, ResTime_block, ResTime_block_v2, ResTime_block_DG, DifferenceConv, GradientConv
from networks.layers.TimeAtt_v1 import MultiheadTimeAttention_v1, MultiheadTimeAttention_v2
import numpy as np



class generator(nn.Module):
    def __init__(self, num_classes, seqlen=100):
        super(generator, self).__init__()
        self.conv_in = nn.Sequential(DifferenceConv(in_channels=1, out_channels=8, kernel_size=(5,7,7), stride=(1,1,1), padding=(2,3,3)),
                                     nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResTime_block_DG(8, 16), ResTime_block_DG(16, 32), ResTime_block_DG(32, 32))
        self.TATT = MultiheadTimeAttention_v1(d_model=32, num_head=8, seqlen=seqlen)
        self.conv_out1 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

    def forward(self, seq_imgs):
        ## 视情况将输入拆分成[1-40, 31-70, 61-100]
        seq_feats = self.conv_in(seq_imgs)
        seq_feats = self.layer1(seq_feats)
        # self.conv_in1 + self.layer1 时域感受野为11（上下5帧）

        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)
        seq_feats = self.TATT(seq_feats)
        seq_feats = self.conv_out1(seq_feats)
        seq_midout = self.conv_out2(seq_feats)
        # seq_midseg = torch.sigmoid(seq_midout).squeeze(dim=1)    ## b, t, h, w
        seq_midseg = seq_midout.squeeze(dim=1)    ## b, t, h, w

        return seq_feats, seq_midseg


class g_loss(nn.Module):
    def __init__(self):
        super(g_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, midpred, target):
        loss_mid = self.bce(midpred, target)

        return loss_mid

class g_SoftLoUloss(nn.Module):
    def __init__(self):
        super(g_SoftLoUloss, self).__init__()

    def forward(self, midpred, target):
        smooth = 0.00
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(midpred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss_mid = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss_mid = 1 - torch.mean(loss_mid)

        return loss_mid
