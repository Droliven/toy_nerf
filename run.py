'''
Author: Droliven levondang@163.com
Date: 2023-05-12 21:58:27
LastEditors: Droliven levondang@163.com
LastEditTime: 2023-05-12 22:00:11
FilePath: \nerf_kunkun0w0\run.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-

"""
@author:   levondang
@software: PyCharm
@project:  generative_models
@file:     run.py
@time:     2023/5/12 18:37
"""
# https://colab.research.google.com/drive/1_51bC5d6m7EFU6U_kkUL2lMYehJqc01R?usp=sharing#scrollTo=7w08xG5X7M8m
# https://github.com/kunkun0w0/Clean-Torch-NeRFs
# In this repo, I implement the Hierarchical volume sampling but I have not used coarse-to-fine strategy.


import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import NeRF
from render import sample_rays_np, render_rays, creat_rays_for_batch_train
from render360 import make_mp4

class Runner:
    def __init__(self):

        device = "cuda:0"
        data_dir = r"./"
        # data_dir = r"H:\datas\generative_models\nerf_kunkun0w0"
        # data_dir = r"/home/songbo/server200/datas/nerf_kunkun0w0"
        # data_dir = r"/home/ml_group/songbo/server204/datasets/nerf_kunkun0w0"
        epoch = 1

        save_dir = "./ckpt"
        os.makedirs(save_dir, exist_ok=True)

        torch.manual_seed(999)
        np.random.seed(666)

        #############################
        # load data
        #############################
        data = np.load(os.path.join(data_dir, 'tiny_nerf_data.npz'), allow_pickle=True)

        images = data['images'] # [b, 100, 100, 3], b=106
        c2w = data['poses'] # [b, 4, 4]
        focal = data['focal']
        H, W = images.shape[1:3]
        print("images.shape:", images.shape)
        print("poses.shape:", c2w.shape)
        print("focal:", focal)

        n_train = 100 # 训练过程用100张图片
        test_img, test_pose = images[101], c2w[101]
        images = images[:n_train]
        c2w = c2w[:n_train]

        # plt.imshow(test_img)
        # plt.show()

        rays = creat_rays_for_batch_train(n_train, images, c2w, H, W, focal, device=device) # [n*h*w, 9]

        #############################
        # training parameters
        #############################
        N = rays.shape[0]
        Batch_size = 4096
        iterations = N // Batch_size
        print(f"There are {iterations} batches of rays and each batch contains {Batch_size} rays")

        bound = (2., 6.) # todo:
        N_samples = (64, None) # todo: 两阶段每条光线采样微元数

        #############################
        # test data
        #############################
        test_rays_o, test_rays_d = sample_rays_np(H, W, focal, test_pose) # [h, w, 3]
        test_rays_o = torch.tensor(test_rays_o, device=device) # [h, w, 3]
        test_rays_d = torch.tensor(test_rays_d, device=device) # [h, w, 3]
        test_rgb = torch.tensor(test_img, device=device) # [h, w, 3]

        #############################
        # training
        #############################
        net = NeRF(use_view_dirs=True).to(device)
        optimizer = torch.optim.Adam(net.parameters(), 5e-4)
        mse = torch.nn.MSELoss()

        for e in range(epoch):
            # create iteration for training
            rays = rays[torch.randperm(N), :] # 随机洗牌
            train_iter = iter(torch.split(rays, Batch_size, dim=0))

            # render + mse
            with tqdm(total=iterations, desc=f"Epoch {e + 1}", ncols=100) as p_bar:
                for i in range(iterations):
                    train_rays = next(train_iter) # [b, 9]
                    assert train_rays.shape == (Batch_size, 9)

                    rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1) # [b, 3]
                    rays_od = (rays_o, rays_d)
                    rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=True)

                    loss = mse(rgb, target_rgb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
                    p_bar.update(1)

            with torch.no_grad():
                rgb_list = list()
                for j in range(test_rays_o.shape[0]):
                    rays_od = (test_rays_o[j], test_rays_d[j])
                    rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, use_view=True)
                    rgb_list.append(rgb.unsqueeze(0))
                rgb = torch.cat(rgb_list, dim=0)
                loss = mse(rgb, torch.tensor(test_img, device=device)).cpu()
                psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
                print(f"PSNR={psnr.item()}")


            torch.save({
                "model": net.state_dict(),
                "epoch": e+1,
                "psnr": psnr.item(),
            }, os.path.join(save_dir, f"epo{e+1}.pth"))

            # 360度环拍视频
            make_mp4(net, H, W, focal, device, bound, N_samples, save_dir, f=f"epo{e+1}.mp4")

        print('Done')


if __name__ == '__main__':
    r = Runner()
