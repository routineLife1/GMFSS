import os
import cv2
import torch
import numpy as np
from model.GMFSS import Model
import warnings
warnings.filterwarnings("ignore")

img_first = 'images/001.jpg'
img_second = 'images/002.jpg'
weigth_path = 'weights'
save_dir = 'output'
n_frames = 10

device = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model = Model()
model.load_model(weigth_path, -1)
model.eval()
model.device(device)

def make_inference(I0, I1, n, scale, pred_bidir_flow=False):    
    timesteps = [(i+1) * 1. / (n+1) for i in range(n)]
    return model.inference(I0, I1, timesteps, scale, pred_bidir_flow)

i0 = cv2.imread(img_first)
i1 = cv2.imread(img_second)

# Higher resolutions require more VRAM.(720p 6.4G)
i0 = cv2.resize(i0,(1280,720))
i1 = cv2.resize(i1,(1280,720))

scale = 1.0 # flow scale
pred_bidir_flow = False # Estimate bilateral optical flow at once

#padding frames
h, w, c = i0.shape
tmp = max(32, int(32 / scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
I0 = cv2.resize(i0,(pw,ph))
I1 = cv2.resize(i1,(pw,ph))

I0 = torch.from_numpy(np.transpose(I0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = torch.from_numpy(np.transpose(I1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.

result = make_inference(I0, I1, n_frames, scale, pred_bidir_flow)
for i in range(1,n_frames):
    cv2.imwrite(os.path.join(save_dir,f"{i}.png"),result[i])