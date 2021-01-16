import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
import models

import glob
from time import time
import math
import copy
import cv2
from skimage.measure import compare_ssim as ssim
from utils import *


learning_rate = 1e-4
layer_num = 19
cs_ratio = 20
gpu_list = '0'
test_dir = 'data'
best_psnr = 0

matrix_dir = 'sampling_matrix'
model_dir = './trained_model'
data_dir = 'data'
result_dir = 'result'
test_name = 'Set11'
test_dir = 'data'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (matrix_dir, cs_ratio)
# Initialization Matrix:
Qinit_data = sio.loadmat(Qinit_Name)
Qinit = Qinit_data['Qinit']


Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)

model = models.LPD_Net(layer_num)
model_name = model.name
model_dir = "./%s/CS_%s_layer_%d_ratio_%d_lr_%.4f" % (model_dir, model_name, layer_num, cs_ratio, learning_rate)

model.load_state_dict(torch.load('./%s/net_params.pkl' % (model_dir)))

print("Model:", model_name)
print("layers:", model.LayerNo)

model = model.to(device)
Phi = Phi.to(device)
Qinit = Qinit.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_%s_layer_%d_ratio_%d_lr_%.4f" % (model_dir, model_name, layer_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

test_dir = os.path.join(test_dir, test_name)
filepaths = glob.glob(test_dir + '/*.tif')
result_dir = os.path.join(result_dir, test_name, model_name, 'cs_ratio_' + str(cs_ratio))

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

print("----------------------------------------------------------------------")
print("Testing. CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):
        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        Icol = img2col_py(Ipad, 33).transpose()/255.0

        Img_output = Icol

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
        x_output, loss_layers_sym = model(Phix, Phi, Qinit)

        end = time()

        Prediction_value = x_output[-1].cpu().data.numpy()

        X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] %s:  PSNR is %.2f, SSIM is %.4f" % (img_no+1, ImgNum, imgName, rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        save_path = ".\%s\%s_%s_ratio_%d_psnr_%.2f_ssim_%.4f.png" % (result_dir, imgName.split('\\')[-1], model_name, cs_ratio, rec_PSNR, rec_SSIM)
        # print(save_path)
        cv2.imwrite(save_path, im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('Avg PSNR/SSIM: %.2f/%.4f' % (np.mean(PSNR_All), np.mean(SSIM_All)))
