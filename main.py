import os
os.environ["CUDA_VISIBLE_DEVICES"] = str("2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from run import Runner as RunnerNeRF
from nerf.nerf_kunkun0w0.render360 import VideoMaker


r = RunnerNeRF() # 训练 NeRF
# m = VideoMaker() # 导入预训练模型，渲染多个视角制作视频