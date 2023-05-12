import os

from run import Runner as RunnerNeRF

os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

r = RunnerNeRF()