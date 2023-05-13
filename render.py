# -*- coding: utf-8 -*-

"""
@author:   levondang
@software: PyCharm
@project:  generative_models
@file:     render.py
@time:     2023/5/12 18:40
"""
# https://colab.research.google.com/drive/1_51bC5d6m7EFU6U_kkUL2lMYehJqc01R?usp=sharing#scrollTo=7w08xG5X7M8m
# https://github.com/kunkun0w0/Clean-Torch-NeRFs
# In this repo, I implement the Hierarchical volume sampling but I have not used coarse-to-fine strategy.

import torch
import torch.nn.functional as F
import numpy as np

from model import PE

def creat_rays_for_batch_train(n_train, images, c2w, H, W, focal, device="cuda:0"):
    '''
    给定训练集所有图片、相机参数，计算每个像素光线的光心、光向、颜色
    Args:
        n_train: n
        images: [n, 100, 100, 3]
        c2w: [n, 4, 4]
        H: 100
        W: 100
        focal: 138.88887889922103
        device:

    Returns:

    '''
    print("Process rays data!")
    rays_o_list = list()
    rays_d_list = list()
    rays_rgb_list = list()

    for i in range(n_train):
        img_i = images[i] # [100, 100, 3]
        c2w_i = c2w[i] # [4, 4]
        rays_o, rays_d = sample_rays_np(H, W, focal, c2w_i) # [100, 100, 3], [100, 100, 3], 同一张图片内部所有像素光心相同，光线角度不同

        rays_o_list.append(rays_o.reshape(-1, 3)) # [100*100, 3]
        rays_d_list.append(rays_d.reshape(-1, 3))  # [100*100, 3]
        rays_rgb_list.append(img_i.reshape(-1, 3))  # [100*100, 3]

    rays_o_npy = np.concatenate(rays_o_list, axis=0) # [n*100*100, 3]
    rays_d_npy = np.concatenate(rays_d_list, axis=0)  # [n*100*100, 3]
    rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)  # [n*100*100, 3]
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=device) # [n*h*w, 9]
    return rays


def sample_rays_np(H, W, f, c2w):
    '''
    给定一张图片的相机位姿和焦距，计算所有像素光线的光心、光向。同一张图片内部所有像素光心相同，光线角度不同
    Args:
        H: 100
        W: 100
        f: 138.88887889922103
        c2w: [4, 4]

    Returns:

    '''
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # 列优先
    dirs = np.stack([(i - W * .5) / f, -(j - H * .5) / f, -np.ones_like(i)], -1) # [100, 100, 3]
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1) # [100, 100, 3]
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d)) # [100, 100, 3]
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf_point(bins, weights, N_samples, device):
    '''
    依据光线的积分系数 -> 概率密度，再基于概率密度有侧重性地采样一些重要位置的微元
    Args:
        bins: uniformN - 1
        weights: [b, uniformN - 2]
        N_samples: 第二阶段采样数目
        device:

    Returns:

    '''
    pdf = F.normalize(weights, p=1, dim=-1) # 概率密度 [b, bins-1]
    cdf = torch.cumsum(pdf, -1) # 累积分布函数 [b, bins-1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # [b, bins]

    # uniform sampling
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device).contiguous() # [b, importantN]

    # invert
    ids = torch.searchsorted(cdf, u, right=True) # [b, importantN] # todo:
    below = torch.max(torch.zeros_like(ids - 1, device=device), ids - 1) # [b, importantN]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids, device=device), ids) # [b, importantN]
    ids_g = torch.stack([below, above], -1) # [b, importantN, 2]
    # ids_g => (batch, N_samples, 2)

    # matched_shape : [batch, N_samples, bins]
    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]] # [b, importantN, bins]
    # gather cdf value
    cdf_val = torch.gather(cdf.unsqueeze(1).expand(matched_shape), -1, ids_g) # [b, importantN, 2], todo:
    # gather z_val
    bins_val = torch.gather(bins[None, None, :].expand(matched_shape), -1, ids_g) # [b, importantN, 2]

    # get z_val for the fine sampling
    cdf_d = (cdf_val[..., 1] - cdf_val[..., 0]) # [b, importantN]
    cdf_d = torch.where(cdf_d < 1e-5, torch.ones_like(cdf_d, device=device), cdf_d)
    t = (u - cdf_val[..., 0]) / cdf_d # [b, importantN]
    samples = bins_val[..., 0] + t * (bins_val[..., 1] - bins_val[..., 0]) # [b, importantN]

    return samples


def uniform_sample_point(tn, tf, N_samples, device):
    '''
    在光线远近区间，按照均匀分布在 N 个 Bin 中随机采样 N 个点
    Args:
        tn: 2.
        tf: 6.
        N_samples: 64
        device: cuda:0

    Returns:

    '''
    k = torch.rand([N_samples], device=device) / float(N_samples) # 均匀分布，[0, 1/N]
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1] # 闭区间采样，共 N 个间隔 -> 去掉1个右区间。剩下 N 个左区间。
    pt_value += k # 相当于在 N 个 Bins 中按均匀分布采样， [N]
    return tn + (tf - tn) * pt_value



def get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=.0, use_view=True):
    '''
    给定一个 Batch 的位置、光向、z, 计算颜色与其积分系数
    Args:
        net: NeRF
        pts: [n, 3]
        rays_d:
        z_vals:
        device:
        noise_std:
        use_view:

    Returns:

    '''
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_d => tensor(Batch_Size, 3)
    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    pts_flat = PE(pts_flat, L=10) # [b*N, (2*10)*3]
    dir_flat = None
    if use_view:
        dir_flat = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), p=2, dim=-1)
        dir_flat = PE(dir_flat, L=4) # [b*N, (2*4)*3]

    rgb, sigma = net(pts_flat, dir_flat) # 输入一个 Batch 的随机均匀采样的 体素位置和光向，输出体素颜色、体素密度
    rgb = rgb.view(list(pts.shape[:-1]) + [3]) # [b, N, 3]
    sigma = sigma.view(list(pts.shape[:-1])) # [b, N]

    # get the interval
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    INF = torch.ones(delta[..., :1].shape, device=device).fill_(1e10)
    delta = torch.cat([delta, INF], -1)
    delta = delta * torch.norm(rays_d, dim=-1, keepdim=True)

    # add noise to sigma
    if noise_std > 0.:
        sigma += torch.randn(sigma.size(), device=device) * noise_std

    # get weights
    alpha = 1. - torch.exp(-sigma * delta)
    ones = torch.ones(alpha[..., :1].shape, device=device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1. - alpha], dim=-1), dim=-1)[..., :-1]

    return rgb, weights


def render_rays(net, rays, bound, N_samples, device, noise_std=.0, use_view=False):
    '''
    给定一个 batch 的打乱的光心、光向，计算积分渲染之后的最终颜色
    Args:
        net: NeRF
        rays: (光心：[b, 3], 光线方向： [b, 3])
        bound: [2., 6.]
        N_samples: 两阶段每条光线采样微元数 (64, None)
        device:
        noise_std:
        use_view: True

    Returns:

    '''
    rays_o, rays_d = rays
    bs = rays_o.shape[0]
    near, far = bound
    uniform_N, important_N = N_samples
    z_vals = uniform_sample_point(near, far, uniform_N, device) #  在光线起止位置之间，按照均匀分布随机采样 N 个点的位置

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None] # [b, sampleN, 3]
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_o, rays_d => tensor(Batch_Size, 3)

    # Run network
    if important_N is not None:
        with torch.no_grad():
            rgb, weights = get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view) # [b, uniformN, 3], [b, uniformN]
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            samples = sample_pdf_point(z_vals_mid, weights[..., 1:-1], important_N, device) # [b, importantN]

        z_vals = z_vals.unsqueeze(0).expand([bs, uniform_N])
        z_vals, _ = torch.sort(torch.cat([z_vals, samples], dim=-1), dim=-1) # [b, uniformN+importantN]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None] # [b, uniformN+importantN, 3]

    # todo: 体素微元太多的话，这行 CUDA 过不了
    rgb, weights = get_rgb_w(net, pts, rays_d, z_vals, device, noise_std=noise_std, use_view=use_view)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map