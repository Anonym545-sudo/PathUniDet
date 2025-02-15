import os
import cv2
import h5py
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

import statistics

from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from Universal import DLASeg
from cross_entropy import cross_entropy_loss
from CE_and_DICE import cross_and_dice_loss,CE_loss
from bounding_losses import FocalLoss,RegL1Loss
from DoDnet import Universal_model



from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

import gc
import h5py
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

def padImage(img, target_height, target_width):
    _, h, w = img.shape

    # 计算垂直方向的填充量，如果差值为奇数，底部会多1
    pad_top = max(0, (target_height - h) // 2)
    pad_bottom = max(0, target_height - h - pad_top)

    # 计算水平方向的填充量，如果差值为奇数，右边会多1
    pad_left = max(0, (target_width - w) // 2)
    pad_right = max(0, target_width - w - pad_left)

    # 使用填充量对图像进行填充
    padded_img = np.pad(img, ((0, 0),(pad_top, pad_bottom), (pad_left, pad_right) ), mode='constant')

    return padded_img


def slideCropF1(img, mask,num_class,task_id):
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]
    if img.shape[1] >= 512 and img.shape[2] >= 512:  # H>512,W>512
        out = normSlideCropF1(img, mask,num_class,task_id)
    elif img.shape[1] >= 512 and img.shape[2] < 512:  # H>=512,W<512
        out = paddingSlideCropF1_1(img, mask,num_class,task_id)
    elif img.shape[1] < 512 and img.shape[2] >= 512:  # H<512,W>=512
        out = paddingSlideCropF1_2(img, mask,num_class,task_id)
    if img.shape[1] < 512 and img.shape[2] < 512:  # H<512,w<512
        out = paddingSlideCropF1_3(img, mask,num_class,task_id)
    
    return out


def slideCropF1(img, mask,num_class,task_id):
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]
    if img.shape[1] >= 512 and img.shape[2] >= 512:  # H>512,W>512
        out = normSlideCropF1(img, mask,num_class,task_id)
    elif img.shape[1] >= 512 and img.shape[2] < 512:  # H>=512,W<512
        out = paddingSlideCropF1_1(img, mask,num_class,task_id)
    elif img.shape[1] < 512 and img.shape[2] >= 512:  # H<512,W>=512
        out = paddingSlideCropF1_2(img, mask,num_class,task_id)
    if img.shape[1] < 512 and img.shape[2] < 512:  # H<512,w<512
        out = paddingSlideCropF1_3(img, mask,num_class,task_id)
    # print('------')
    # print(f1_score)
    # print(iou_score)
    return out


def normSlideCropF1(img, mask,num_class,task_id):  # H>512,W>512
    # total_out=torch.zeros(img.shape)
    # print(img.shape)
    _, H, W = img.shape
    total_out=torch.zeros(1,num_class,H, W)
    m = H // 512  # img在width能整切多少
    n = W // 512  # img在height能整切多少
    h_reminder = H % 512
    w_reminder = W % 512  # 以 3044 x 2018 为例，m=5,n=3,h_r=484,w_r=482
    
    stride = 512
    TP = 0
    FP = 0
    FN = 0

    for i in range(0, m + 1):  # height。 0、1、2、3、4是整切块，5是边缘切块
        for j in range(0, n + 1):  # width。  0、1、2是整切块，3是边缘切块
            # 1.先切割：分4种情况
            # 2.输入patch，得到 out
            # 3.比较 out与 mask，计算TP、FP、FN
            if i < m and j < n:
                patch = img[:, i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]
                patch_mask = mask[:, i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]

                # patch = torch.Tensor(patch)
                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.unsqueeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上
                # DoDnet:
                patch=patch.cuda()
                with torch.no_grad():
                    out = unet.forward(patch,task_id)
                out=out[:,:num_class]                
                # out=out.squeeze(4)
                out=F.softmax(out,dim=1)

                total_out[:, :,i * stride:(i + 1) * stride, j * stride:(j + 1) * stride]=out

                

            if w_reminder != 0 and j == n and i < m:  # 当行不能被整切时，才从最后重合切一块
                patch = img[:, i * stride:(i + 1) * stride, W - stride:W]  # patch需要 512 x 512
                patch_mask = mask[:, i * stride:(i + 1) * stride, W - w_reminder:W]  # mask保持不变

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.unsqueeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch=patch.cuda()
                with torch.no_grad():
                    out = unet.forward(patch,task_id)
                out=out[:,:num_class]
                
                # out=out.squeeze(4)
                out=F.softmax(out,dim=1)
                out = out[0:1]
                out = out[:,:, :, 512 - w_reminder:512]  # 裁剪成与mask一致的大小、位置
                total_out[:, :,i * stride:(i + 1) * stride, W - w_reminder:W]=out
               

            if h_reminder != 0 and i == m and j < n:  # 当列不能被整切时，才从最后重合切一块
                patch = img[:, H - stride:H, j * stride:(j + 1) * stride]
                patch_mask = mask[:, H - h_reminder:H, j * stride:(j + 1) * stride]

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.unsqueeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch=patch.cuda()
                with torch.no_grad():
                    out = unet.forward(patch,task_id)
                out=out[:,:num_class]
                # out=out.squeeze(4)
                out=F.softmax(out,dim=1)
                out = out[:, :, 512 - h_reminder:512, :]  # 裁剪成与mask一致的大小、位置
                total_out[:, :,H - h_reminder:H, j * stride:(j + 1) * stride]=out

                

            if (w_reminder != 0 and h_reminder != 0) and i == m and j == n:  
                patch = img[:, H - stride:H, W - stride:W]
                patch_mask = mask[:, H - h_reminder:H, W - w_reminder:W]
                # print(patch.shape)

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.unsqueeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch=patch.cuda()
                with torch.no_grad():
                    out = unet.forward(patch,task_id)
                out=out[:,:num_class]
                # out=out.squeeze(4)
                out=F.softmax(out,dim=1)
                out = out[:, :, 512 - h_reminder:512, 512 - w_reminder:512]  # 裁剪成与mask一致的大小、位置

                total_out[:, :,H - h_reminder:H, j * stride:(j + 1) * stride]=out

                out = out.cpu()
                # print(out.shape)
                

            # print(i, j)
            # print("TP:", TP)
            # print("FP:", FP)
            # print("FN:", FN)

    # precision,recall,f1_score,iou_score = calculate_f1_score(TP, FP, FN)
    return total_out


def paddingSlideCropF1_1(img, mask,num_class,task_id):  # H>=512,W<512
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    paddingLen = 512 - W  # 在W上填充了多少
    half_r = paddingLen // 2

    m = H // 512
    reminder = H % 512
    img_padding = padImage(img, H, 512)  # 填充为 H x 512大小

    total_out=torch.zeros(1,num_class,H, W)

    for i in range(0, m + 1):
        if i < m:
            patch = img_padding[:, i * 512:(i + 1) * 512, :]
            patch_mask = mask[:, i * 512:(i + 1) * 512, :]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.unsqueeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch=patch.cuda()
            with torch.no_grad():
                out = unet.forward(patch,task_id)
            out=out[:,:num_class]
            # out=out.squeeze(4)
            out=F.softmax(out,dim=1)
            out = out[:, :, :, half_r:half_r + W]  # 将padding后的部分切除再比较
            total_out[:, :,i * 512:(i + 1) * 512, :]=out


        if reminder != 0 and i == m:
            patch = img_padding[:, H - 512:H, :]
            patch_mask = mask[:, H - reminder:H, :]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.unsqueeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch=patch.cuda()
            with torch.no_grad():
                out = unet.forward(patch,task_id)
            out=out[:,:num_class]
            # out=out.squeeze(4)
            out=F.softmax(out,dim=1)
            out = out[:, :, 512 - reminder:512, half_r:half_r + W]  # 将padding后的部分、重合部分切除再比较

            total_out[:,:, H - reminder:H, :]=out

        # print(i)
        # print("TP:", TP)
        # print("FP:", FP)
        # print("FN:", FN)

    # precision,recall,f1_score,iou_score = calculate_f1_score(TP, FP, FN)
    return total_out


def paddingSlideCropF1_2(img, mask,num_class,task_id):  # H<512 , W>=512
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    paddingLen = 512 - H  # 在H上填充了多少
    half_r = paddingLen // 2

    m = W // 512
    reminder = W % 512
    img_padding = padImage(img, 512, W)

    total_out=torch.zeros(1,num_class,H, W)

    for i in range(0, m + 1):
        if i < m:
            patch = img_padding[:, :, i * 512:(i + 1) * 512]
            patch_mask = mask[:, :, i * 512:(i + 1) * 512]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.unsqueeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch=patch.cuda()
            with torch.no_grad():
                out = unet.forward(patch,task_id)
            out=out[:,:num_class]
            # out=out.squeeze(4)
            out=F.softmax(out,dim=1)
            out = out[:, :, half_r:half_r + H, :]  # 将padding后的部分切除再比较

            total_out[:, :, :, i * 512:(i + 1) * 512]=out
            

        if reminder != 0 and i == m:
            patch = img_padding[:, :, W - 512:W]
            patch_mask = mask[:, :, W - reminder:W]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.unsqueeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            with torch.no_grad():
                out = unet.forward(patch,task_id)
            out=out[:,:num_class]
            # out=out.squeeze(4)
            out=F.softmax(out,dim=1)
            out = out[:, :, half_r:half_r + H, 512 - reminder:512]  # 将padding后的部分、重合部分切除再比较

            total_out[:, :, :, W - reminder:W]=out



    return total_out

def paddingSlideCropF1_3(img, mask,num_class,task_id):
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    halfPadLen_H = (512 - H) // 2
    halfPadLen_W = (512 - W) // 2

    total_out=torch.zeros(1,num_class,H, W)
    
    imgShape=img.shape

    # print("imgShape:",imgShape)

    patch = img
    patch_mask = mask

    patch = padImage(patch, 512, 512)

    patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
    patch = patch.unsqueeze(0)  # 1,3,512,512
    patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上
    patch=patch.cuda()
    with torch.no_grad():
        out = unet.forward(patch,task_id)
    out=out[:,:num_class]
    # out=out.squeeze(4)
    out=F.softmax(out,dim=1)
    out = out[:, :, halfPadLen_H:halfPadLen_H + H, halfPadLen_W:halfPadLen_W + W]  # 将padding后的部分、重合部分切除再比较
    total_out=out
    
    
    return total_out



model_path = '/media/ipmi2022/unet_down_time_newdata_2024-10-17_04_28_33_epoch_2000.pkl'
# model_path='/media/ipmi2022/Elements/backup/xuzhengyang/code/universal_model/universal_segmantation/unet_wodown_time_newdata_2024-12-02_05_02_51_final.pkl'
model_state_dict = torch.load(model_path)
# unet=Universal_model().cuda()
heads={'seg':15,'kp':4 ,'hm': 4, 'wh': 2, 'reg': 2}
unet=DLASeg('dla60', heads,
                 pretrained=True,
                 down_ratio=2,
                 head_conv=256).cuda()
unet.load_state_dict(model_state_dict)
unet.eval()

data_folder = '/media/ipmi2022/SCSI_all/xuzhengyang/universal_test'
# task_id_dict={'lizard':0, 'TNBC':1, 'cpm15':2, 'GLAS':3, 'cpm17':4, 'MonuSeg':5, 'PanNuke':6, 'Kumar':7, 'ConSep':8, 'Her2':9, 'CRAGv2':10}
# task_id_dict = {'lizard':0,'TNBC':1,'cpm15':2,'GLAS':3,'cpm17':4,'MonuSeg':5,'Kumar':6,'ConSep':7,'Her2':8,'CRAGv2':9}
# channel_dict={'lizard':7,'TNBC':2,'cpm15':2,'GLAS':2,'cpm17':2,'MonuSeg':2,'PanNuke':6,'Kumar':2,'ConSep':8,'Her2':3,'CRAGv2':2}
datasets=['EndoNuke','OCELOT','RINGS','TNBC','PDL1','Kumar','CryoNuSeg']
task_id_dict={'OCELOT':0,'RINGS':1,'TNBC':4,'PDL1':3,'Kumar':5,'CryoNuSeg':5,'EndoNuke':6}
channel_dict={'OCELOT':2,'RINGS':2,'TNBC':2,'PDL1':3,'Kumar':2,'CryoNuSeg':2,'EndoNuke':4}




# for dataset in os.listdir(data_folder):
for dataset in datasets:
    iou_list=[]
    f1_list=[]
    precision_list=[]
    recall_list=[]


    task_id=task_id_dict[dataset]
    channel=channel_dict[dataset]
    total_F1=torch.zeros(channel)
    total_IOU=torch.zeros(channel)
    total_precision=torch.zeros(channel)
    total_recall=torch.zeros(channel)
    img_folder=os.path.join(data_folder,dataset,'img')
    # print(img_folder)
    hist = torch.zeros(channel, channel).cuda()
    cnt=0

    for folder1 in os.listdir(img_folder):
        

        img_path=os.path.join(img_folder,folder1)

        # 提取图像ID和数据集名称
        folder, num_id = os.path.split(img_path)
        num_id = num_id.replace('.jpg', '').replace('.png', '').replace('.tif', '')
        folder, _ = os.path.split(folder)
        _, dataset_name = os.path.split(folder)
        _, pkl_time = os.path.split(model_path)
        pkl_time = pkl_time.replace('.pkl', '')

        # 构建结果保存路径
        store_name = f'{pkl_time}_datasetname_{dataset_name}_img_id_{num_id}.h5'
        # 构建掩码路径
        h5_path = img_path.replace('img', 'ground_truth').replace('jpg', 'h5').replace('tif', 'h5').replace('png', 'h5')
        # 从HDF5文件中读取掩码数据
        with h5py.File(h5_path, 'r') as hf:
            # 读取掩码数据为numpy数组
            heatmap = np.array(hf.get('heatmap'))

        # 获取掩码的类别数和尺寸信息
        num_class, _, _ = heatmap.shape

        # 读取图像
        images = Image.open(img_path)

        # 进行图像预处理
        # 将图像转换为Tensor
        images = transforms(images)
        # 获取图像的尺寸信息
        _, H, W = images.shape
        # 对图像和掩码进行裁剪
        
        out = slideCropF1(images, heatmap,num_class,task_id)
        out=torch.argmax(out, dim=1).cuda()
        heatmap=torch.from_numpy(heatmap)
        heatmap=heatmap.unsqueeze(0)
        heatmap=torch.argmax(heatmap, dim=1).cuda()
        heatmap=heatmap.view(-1)
        out=out.view(-1)
        hist_one_epoch = torch.bincount(
                heatmap * num_class + out,
                minlength=num_class ** 2
                ).view(num_class, num_class)
        hist+=hist_one_epoch
        ious_one_epoch = hist_one_epoch.diag() / (hist_one_epoch.sum(dim=0) + hist_one_epoch.sum(dim=1) - hist_one_epoch.diag()+1e-6)
        iou_list.append(torch.mean(ious_one_epoch).cpu().numpy().item())
    ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        
    print(torch.mean(ious))
    print("{} :total_var_IOU is {:.4f}".format(dataset,statistics.variance(iou_list)))
