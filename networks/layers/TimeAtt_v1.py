import math
import torch
import torch.nn as nn
from typing import List
from torch.nn import init
import torch.nn.functional as F


class MultiheadTimeAttention_v1(nn.Module):
    def __init__(self, d_model=32, num_head=8, seqlen=100, bias=True):
        super(MultiheadTimeAttention_v1, self).__init__()

        self.seqlen = seqlen
        self.num_head = num_head
        self.hidden_dim = d_model // num_head

        QK_heads: List[nn.Module] = []
        for i in range(num_head):
            QK_heads.append(nn.Linear(seqlen, seqlen))

        self.QK_heads = nn.Sequential(*QK_heads)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm3d(d_model)
        self.conv = nn.Sequential(nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(1,1,1), stride=(1,1,1)),
                                  nn.BatchNorm3d(d_model), nn.ReLU(inplace=True))   ## 相当于projection



    def forward(self, input):  # size: (1, 512, 512, 8*4, 100)
        bs, h, w, d, slen = input.size()
        value = input / (self.seqlen**0.5)
        value = value.view(bs, h, w, self.num_head, self.hidden_dim, slen)

        qkv = torch.zeros_like(value).to(input.device)
        for i in range(self.num_head):
            qkv[:,:,:,i,:,:] = self.QK_heads[i](value[:,:,:,i,:,:])

        qkv = qkv.view(bs, h, w, -1, slen).permute(0, 3, 4, 1, 2)
        qkv = self.relu(self.norm1(qkv))
        x = self.conv(qkv)
        return x
