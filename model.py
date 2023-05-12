# -*- coding: utf-8 -*-

"""
@author:   levondang
@software: PyCharm
@project:  generative_models
@file:     model.py
@time:     2023/5/12 18:39
"""
# https://colab.research.google.com/drive/1_51bC5d6m7EFU6U_kkUL2lMYehJqc01R?usp=sharing#scrollTo=7w08xG5X7M8m
# https://github.com/kunkun0w0/Clean-Torch-NeRFs
# In this repo, I implement the Hierarchical volume sampling but I have not used coarse-to-fine strategy.

import torch.nn as nn
import torch
import torch.nn.functional as F

# Positional encoding (section 5.1)
def PE(x, L):
    '''

    Args:
        x: [b, 3]
        L: 10 / 4

    Returns:

    '''
    pe = list()
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2. ** i * x))
    return torch.cat(pe, -1) # [b, 2*(10/4)*3] = [b, 60/24]


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_views=24, skip=4, use_view_dirs=True):
        """
        :param D: depth of MLP backbone
        :param W: width of MLP backbone
        :param input_ch: encoded RGB input's channels
        :param input_ch_views: encoded view input's channels
        :param skip: when skip connect
        :param use_view_dirs: use view-dependent
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skip = skip
        self.use_view_dirs = use_view_dirs

        self.net = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D-1):
            if i == skip:
                self.net.append(nn.Linear(W + input_ch, W))
            else:
                self.net.append(nn.Linear(W, W))

        self.sigma_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        if use_view_dirs:
            self.proj = nn.Linear(W + input_ch_views, W // 2)
        else:
            self.proj = nn.Linear(W, W // 2)

        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_pts, input_views=None):
        '''
        输入一个 Batch 的体素位置和光向，输出体素颜色、体素密度
        Args:
            input_pts: 光心，[b, 60]
            input_views: 光向，[b, 24]

        Returns:

        '''
        h = input_pts.clone()
        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        sigma = F.relu(self.sigma_linear(h))
        feature = self.feature_linear(h)

        if self.use_view_dirs:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, sigma # [b, 3], [b, 1]