import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
import torch.nn.functional as F
from MIMSN import MIMSN,_initialize_weights
from PIL import Image

# 设置随机数种子
seed = 99
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#定义TV损失和LOSS
def tv_regularization(img):
    dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)

def loss_fn(y, y_pred):
    return torch.mean(torch.square(y - y_pred))

def loss_ssim(image1,image2):
    return pytorch_ssim.ssim(image1,image2) 

def psnr(target, prediction):
    mse = F.mse_loss(target, prediction) 
    max_pixel = torch.max(target)  
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr

def binarize_matrix(matrix, threshold):
    rows = len(matrix)
    cols = len(matrix[0])
    binary_matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] >= threshold:
                binary_matrix[i][j] = 255
            else:
                binary_matrix[i][j] = 0
    return binary_matrix

#选择GPU进行训练
GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 
    dtype = torch.cuda.FloatTensor 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    print("num GPUs",torch.cuda.device_count()) 
else:
    dtype = torch.FloatTensor 
device=torch.device('cuda:0' if GPU else 'cpu')

#定义参数
img_W = 64
img_H = 64
SR = 500/(64*64)                     # sampling rate
batch_size = 1                                
lr0 = 0.05                           # learning rate
TV_strength = 1e-9                            # regularization parameter of Total Variation
num_patterns = int(np.round(img_W*img_H*SR))  # number of measurement times  
Steps = 501                                 # optimization steps                              
decay_rate = 0.90
decay_steps = 100
alpha = 0.2
alpha_step = 0.001

result_save_path = 'result/'
patterns = np.load('patterns\ss_50.npy') 
y = np.loadtxt(r'data\\50.txt')   

# DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0 
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = patterns[i,:,:] 
    B_r = y[i]  
    count = count + 1
    SI_aver = (SI_aver * (count -1) + pattern * B_r)/count 
    B_aver  = (B_aver * (count -1) + B_r)/count 
    R_aver = (R_aver * (count -1) + sum(sum(pattern)))/count 
    RI_aver = (RI_aver * (count -1) + sum(sum(pattern))*pattern)/count 
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
print('Finished')

y_real = torch.tensor(y[0:num_patterns])
A_real = torch.tensor(patterns[0:num_patterns,:,:]).float().to(device)
y_real = torch.reshape(y_real,(1,num_patterns,1,1)).float().to(device)
mean_ya, variance_ya = torch.mean(y_real), torch.var(y_real)
y_real = (y_real - mean_ya) / torch.sqrt(variance_ya)
mean_a, variance_a = torch.mean(A_real), torch.var(A_real)
A_real = (A_real - mean_a) / torch.sqrt(variance_a)

model = MIMSN().cuda()
_initialize_weights(model)
l1_regularization = 0
l2_regularization = 0
for param in model.parameters():
    l1_regularization += torch.abs(param).sum()
    l2_regularization += (param ** 2).sum()

DGI = (DGI - np.mean(DGI))/np.std(DGI) 
DGI = np.reshape(DGI,[img_W,img_H],order='F') 
DGI = torch.tensor(DGI)
input2 = torch.reshape(DGI,(1,1,img_W,img_H)).to(device).float()
input2 = torch.randn(1,1,img_W,img_H).to(device).float()
fc_layer = nn.Linear(num_patterns, img_H*img_W).to(device)
input1 = torch.reshape(y_real,(1,1,1,num_patterns)).float().to(device)
input1 = fc_layer(input1)
input1 = torch.reshape(input1,(1,1,img_W,img_H)).to(device).float()

parameters = list(model.parameters()) + list(fc_layer.parameters())
optimizer = optim.Adam(parameters,lr=lr0,betas=(0.5, 0.999),eps=1e-08)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate, last_epoch=-1, verbose=False)

for epoch in range(Steps):
    model.train()
    x_pred1,y_pred1 = model(input1.detach(),A_real,img_W,img_H,num_patterns)
    x_pred2,y_pred2 = model(input2.detach(),A_real,img_W,img_H,num_patterns)
    x_pred = (x_pred1+x_pred2)/2
    TV_reg = TV_strength *tv_regularization(x_pred1)
    loss_y1 = loss_fn(y_real, y_pred1) 
    loss1 = loss_y1+TV_reg 
    TV_reg = TV_strength *tv_regularization(x_pred2)
    loss_y2 = loss_fn(y_real, y_pred2) 
    loss2 = loss_y2+TV_reg
    loss_diff = torch.mean(torch.abs(x_pred1-x_pred2))
    loss = alpha * loss_diff + (1-alpha)*(loss1 + loss2)
    if alpha >= 0.8:
        loss = 0.8 * loss_diff + (1-alpha)*(loss1 + loss2)
    else:
        alpha = alpha + alpha_step

    #反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%50 == 0: 
            print('step:%d----y loss:%f:' % (epoch,loss)) 
            DGI_temp0 = np.reshape(DGI,(img_H,img_W)) 
            x_pred = x_pred - torch.min(x_pred)
            x_pred = x_pred*255/torch.max(torch.max(x_pred))
            x_pred = torch.reshape(x_pred,(img_H,img_W))
            x_pred = Image.fromarray(x_pred.detach().cpu().numpy().astype('uint8')).convert('L')
            x_pred.save(result_save_path + 'E_%d_%d.bmp'%(num_patterns,epoch))

            DGI_temp0 = DGI_temp0 - torch.min(DGI_temp0)
            DGI_temp0 = DGI_temp0*255/torch.max(torch.max(DGI_temp0))
            DGI_temp0 = torch.reshape(DGI_temp0,(img_H,img_W))
            DGI_temp0 = Image.fromarray(DGI_temp0.detach().cpu().numpy().astype('uint8')).convert('L')
            DGI_temp0.save(result_save_path + 'E_DGI_%d.bmp'%(num_patterns))
