# -*- coding: utf-8 -*-

"""
@author:   levondang
@software: PyCharm
@project:  generative_models
@file:     render360.py
@time:     2023/5/12 21:01
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
import torch
import imageio
from tqdm import tqdm

from render import sample_rays_np, render_rays
from model import NeRF


def pose_spherical(theta, phi, radius, trans_t, rot_phi, rot_theta):
    '''

    Args:
        theta: 0.0-360.0
        phi: -30.0
        radius: 4.0
        trans_t:
        rot_phi:
        rot_theta:

    Returns:

    '''
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w # [4, 4]
    return c2w



def make_mp4(net, H, W, focal, device, bound, N_samples, out_dir, f = 'video.mp4'):
    trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=float)

    rot_phi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=float)

    rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=float)

    frames = []
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        with torch.no_grad():
            c2w = pose_spherical(th, -30., 4., trans_t, rot_phi, rot_theta) # [4, 4] 
            rays_o, rays_d = sample_rays_np(H, W, focal, c2w) # [h, w, 3] 
            # 逐行渲染，避免显卡过载
            rgb = []
            for i in range(H):
                row_rays_o = rays_o[i] # [w, 3]
                row_rays_d = rays_d[i] # [w, 3]
                row_rays_od = (torch.tensor(row_rays_o, device=device, dtype=torch.float32),
                           torch.tensor(row_rays_d, device=device, dtype=torch.float32))
                row_rgb, row_depth, row_acc = render_rays(net, row_rays_od, bound=bound, N_samples=N_samples, device=device, use_view=True)

                rgb.append(row_rgb.cpu().data.numpy())

        frames.append((255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8))

    imageio.mimwrite(os.path.join(out_dir, f), frames, fps=30, quality=7) # 需要 pip install imageio[ffmpeg]


class VideoMaker:
    def __init__(self):
        device = "cuda:0"
        # data_dir = r"H:\datas\generative_models\nerf_kunkun0w0"
        # data_dir = r"/home/songbo/server200/datas/nerf_kunkun0w0"
        data_dir = r"/home/ml_group/songbo/server204/datasets/nerf_kunkun0w0"

        save_dir = "./ckpt"
        os.makedirs(save_dir, exist_ok=True)

        #############################
        # load data
        #############################
        data = np.load(os.path.join(data_dir, 'tiny_nerf_data.npz'), allow_pickle=True)

        images = data['images']  # [b, 100, 100, 3], b=106
        c2w = data['poses']  # [b, 4, 4]
        focal = data['focal']
        H, W = images.shape[1:3]
        print("images.shape:", images.shape)
        print("poses.shape:", c2w.shape)
        print("focal:", focal)

        bound = (2., 6.)  # todo:
        # N_samples = (64, None) # todo: 两阶段每条光线采样微元数
        N_samples = (192, 64)  # todo: 两阶段每条光线采样微元数

        print(f"rays between: [{bound[0]}, {bound[1]}], volumes num: [{N_samples[0]}, {N_samples[1]}]")

        net = NeRF(use_view_dirs=True).to(device)
        state = torch.load(os.path.join(save_dir, f"epo{50}.pth"), map_location=device)
        net.load_state_dict(state["model"])

        net.eval()

        make_mp4(net, H, W, focal, device, bound, N_samples, save_dir, f=f"epo{-1}.mp4")


if __name__ == '__main__':
    m = VideoMaker()
