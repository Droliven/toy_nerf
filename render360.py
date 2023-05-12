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
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
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
            c2w = pose_spherical(th, -30., 4., trans_t, rot_phi, rot_theta)
            rays_o, rays_d = sample_rays_np(H, W, focal, c2w[:3, :4])
            rays_od = (torch.tensor(rays_o, device=device, dtype=torch.float32),
                       torch.tensor(rays_d, device=device, dtype=torch.float32))
            rgb, depth, acc = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=True)

        frames.append((255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8))

    imageio.mimwrite(os.path.join(out_dir, f), frames, fps=30, quality=7) # 需要 pip install imageio[ffmpeg]


if __name__ == '__main__':
    device = "cuda:0"
    data_dir = r"./"
    # data_dir = r"H:\datas\generative_models\nerf_kunkun0w0"
    save_dir = "./"
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
    N_samples = (64, None)  # todo: 两阶段每条光线采样微元数


    net = NeRF(use_view_dirs=True).to(device)
    state = torch.load(os.path.join(save_dir, f"epo{9}.pth"), map_location=device)
    net.load_state_dict(state["model"])

    net.eval()

    make_mp4(net, H, W, focal, device, bound, N_samples, save_dir, f=f"epo{-1}.mp4")