# NeRF Tiny demo
> 一个简化版的 NeRF 实现，包括位置编码、粗糙采样+精细采样等主要设计要素，但没有分两个阶段。主要用于极快地训练模型并梳理 NeRF 的主要原理，参考自[kunkun0w0](https://github.com/kunkun0w0/Clean-Torch-NeRFs)。

+ 重新训练 NeRF: `python main.py`
+ 预训练模型 `pretrained/epo50_bs1024_samples192-64.pth`及360环绕渲染结果：`pretrained/epo50.mp4`
