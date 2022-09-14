import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from pandas import DataFrame
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.lpips import LPIPS

#model
from model.GMFSS import Model

#train_log path
log_path = 'weights'

# path
abspath = os.path.abspath(__file__)
app_dir = os.path.dirname(abspath)
path = 'dataset/'
f = list()
for d in os.listdir(path):
    d = os.path.join(app_dir, path, d)
    if os.path.isdir(d):
        f.append(d)

# initiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_model(log_path, -1)
model.eval()
model.device()
lpips = LPIPS(net='alex').to(device)

# bench it out!
loss_list = list()
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    # read
    im_path = os.listdir(name)
    I0 = cv2.imread(os.path.join(name, im_path[0]))  # first
    I1 = cv2.imread(os.path.join(name, im_path[len(im_path) // 2]))  # middle, C, H, W
    I2 = cv2.imread(os.path.join(name, im_path[-1]))  # last

    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)  # B, C, H, W
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)  # B, C, H, W
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)  # B, C, H, W

    pad_size = (544, 960)
    I0 = F.interpolate(I0, pad_size, mode="bilinear", align_corners=False)
    I1 = F.interpolate(I1, pad_size, mode="bilinear", align_corners=False)
    I2 = F.interpolate(I2, pad_size, mode="bilinear", align_corners=False)

    # inference and test
    mid = model.inference(I0, I2)  # B, C, H, W

    ssim = ssim_matlab(torch.round(I1 * 255.) / 255., torch.round(mid * 255.) / 255.).detach().cpu().numpy()

    loss_lpips = lpips.forward((mid[0].flip(0) - 0.5) / 0.5, (I1[0].flip(0) - 0.5) / 0.5).mean().detach().cpu().numpy()  # input RGB

    mid = (torch.round(mid[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)  # H, W, C
    I1 = (torch.round(I1[0] * 255.) / 255.).detach().cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())

    loss_list.append({'path': name, 'psnr': psnr, 'ssim': ssim, 'lpips': loss_lpips})

loss_df = DataFrame(loss_list)
loss_df.to_csv('loss.csv')
