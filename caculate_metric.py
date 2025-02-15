import os
import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# from unet5 import Universal_model
# from unet3 import Universal_model
from DoDnet5_image import Universal_model

from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from unet import UNet

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])
unet = Universal_model().cuda()

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    values, indices = torch.max(net_output, dim=1, keepdim=True)
    mask = torch.zeros_like(net_output)
    mask.scatter_(1, indices, 1)
    net_output=mask

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)
    
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn



def calculate_f1_score(TP, FP, FN):

    # TP = TP.item()
    # FP = FP.item()
    # FN = FN.item()

    # print("f1内：",TP)
    # print("f1内：", FP)
    # print("f1内：", FN)

    # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    precision = (TP + 1e-10) / (TP + FP + 1e-10) 
    recall = (TP + 1e-10) / (TP + FN + 1e-10) 

    # print("精度：",precision)
    # print("recall:",recall)

    # 计算 F1 分数
    f1_score = (2 * precision * recall + 1e-10) / (precision + recall+1e-10) 

    iou_score=(2*TP+ 1e-10)/(2*TP+FP+FN+1e-10)
    # print("f1:",f1_score)

    return f1_score,iou_score


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


def slideCropF1(img, mask,num_class):
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]
    if img.shape[1] >= 512 and img.shape[2] >= 512:  # H>512,W>512
        F1,IOU = normSlideCropF1(img, mask,num_class)
    elif img.shape[1] >= 512 and img.shape[2] < 512:  # H>=512,W<512
        F1,IOU = paddingSlideCropF1_1(img, mask,num_class)
    elif img.shape[1] < 512 and img.shape[2] >= 512:  # H<512,W>=512
        F1,IOU = paddingSlideCropF1_2(img, mask,num_class)
    if img.shape[1] < 512 and img.shape[2] < 512:  # H<512,w<512
        F1,IOU = paddingSlideCropF1_3(img, mask,num_class)
    return F1,IOU


def normSlideCropF1(img, mask,num_class):  # H>512,W>512
    # print(img.shape)
    _, H, W = img.shape
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
                patch = patch.squeeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上
                # DoDnet:
                patch = patch.repeat(4, 1, 1, 1).cuda()
                with torch.no_grad():
                    out = unet(patch, task_id, num_class)
                out = out[0:1]  # 送网络后，不用切割，直接比较

                out = out.cpu()
                patch_mask = patch_mask.unsqueeze(0).cpu()

                shp_reg=out.shape
                axes = [0] + list(range(2, len(shp_reg)))
                tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)

                TP += tp
                FP += fp
                FN += fn

            if w_reminder != 0 and j == n and i < m:  # 当行不能被整切时，才从最后重合切一块
                patch = img[:, i * stride:(i + 1) * stride, W - stride:W]  # patch需要 512 x 512
                patch_mask = mask[:, i * stride:(i + 1) * stride, W - w_reminder:W]  # mask保持不变

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.squeeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch = patch.repeat(4, 1, 1, 1).cuda()
                with torch.no_grad():
                    out = unet(patch, task_id, num_class)
                out = out[0:1]
                out = out[:,:, :, 512 - w_reminder:512]  # 裁剪成与mask一致的大小、位置

                out = out.cpu()
                patch_mask = patch_mask.unsqueeze(0).cpu()

                shp_reg=out.shape
                axes = [0] + list(range(2, len(shp_reg)))
                tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


                TP += tp
                FP += fp
                FN += fn

            if h_reminder != 0 and i == m and j < n:  # 当列不能被整切时，才从最后重合切一块
                patch = img[:, H - stride:H, j * stride:(j + 1) * stride]
                patch_mask = mask[:, H - h_reminder:H, j * stride:(j + 1) * stride]

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.squeeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch = patch.repeat(4, 1, 1, 1).cuda()
                with torch.no_grad():
                    out = unet(patch, task_id, num_class)
                out = out[0:1]
                out = out[:, :, 512 - h_reminder:512, :]  # 裁剪成与mask一致的大小、位置

                out = out.cpu()
                patch_mask = patch_mask.unsqueeze(0).cpu()

                shp_reg=out.shape
                axes = [0] + list(range(2, len(shp_reg)))
                tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


                TP += tp
                FP += fp
                FN += fn

            if (w_reminder != 0 and h_reminder != 0) and i == m and j == n:  
                patch = img[:, H - stride:H, W - stride:W]
                patch_mask = mask[:, H - h_reminder:H, W - w_reminder:W]
                # print(patch.shape)

                patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
                patch = patch.squeeze(0)  # 1,3,512,512
                patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

                # DoDnet:
                patch = patch.repeat(4, 1, 1, 1).cuda() # 4,3,512,512
                with torch.no_grad():
                    out = unet(patch, task_id, num_class)
                out = out[0:1]  # 1,3,512,512   送网络后，需要将最后一块的公共部分切割之后比较
                out = out[:, :, 512 - h_reminder:512, 512 - w_reminder:512]  # 裁剪成与mask一致的大小、位置

                out = out.cpu()
                # print(out.shape)
                patch_mask = patch_mask.unsqueeze(0).cpu()

                shp_reg=out.shape
                axes = [0] + list(range(2, len(shp_reg)))
                tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


                TP += tp
                FP += fp
                FN += fn

            # print(i, j)
            # print("TP:", TP)
            # print("FP:", FP)
            # print("FN:", FN)

    f1,iou = calculate_f1_score(TP, FP, FN)
    return f1,iou


def paddingSlideCropF1_1(img, mask,num_class):  # H>=512,W<512
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    paddingLen = 512 - W  # 在W上填充了多少
    half_r = paddingLen // 2

    m = H // 512
    reminder = H % 512
    img_padding = padImage(img, H, 512)  # 填充为 H x 512大小

    for i in range(0, m + 1):
        if i < m:
            patch = img_padding[:, i * 512:(i + 1) * 512, :]
            patch_mask = mask[:, i * 512:(i + 1) * 512, :]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.squeeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch = patch.repeat(4, 1, 1, 1).cuda()
            with torch.no_grad():
                out = unet(patch, task_id, num_class)
            out = out[0:1]
            out = out[:, :, :, half_r:half_r + W]  # 将padding后的部分切除再比较

            out = out.cpu()
            patch_mask = patch_mask.unsqueeze(0).cpu()

            shp_reg=out.shape
            axes = [0] + list(range(2, len(shp_reg)))
            tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


            TP += tp
            FP += fp
            FN += fn

        if reminder != 0 and i == m:
            patch = img_padding[:, H - 512:H, :]
            patch_mask = mask[:, H - reminder:H, :]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.squeeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch = patch.repeat(4, 1, 1, 1).cuda()
            with torch.no_grad():
                out = unet(patch, task_id, num_class)
            out = out[0:1]
            out = out[:, :, 512 - reminder:512, half_r:half_r + W]  # 将padding后的部分、重合部分切除再比较

            out = out.cpu()
            patch_mask = patch_mask.unsqueeze(0).cpu()

            shp_reg=out.shape
            axes = [0] + list(range(2, len(shp_reg)))
            tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


            TP += tp
            FP += fp
            FN += fn

        # print(i)
        # print("TP:", TP)
        # print("FP:", FP)
        # print("FN:", FN)

    f1,iou = calculate_f1_score(TP, FP, FN)
    return f1,iou


def paddingSlideCropF1_2(img, mask,num_class):  # H<512 , W>=512
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    paddingLen = 512 - H  # 在H上填充了多少
    half_r = paddingLen // 2

    m = W // 512
    reminder = W % 512
    img_padding = padImage(img, 512, W)

    for i in range(0, m + 1):
        if i < m:
            patch = img_padding[:, :, i * 512:(i + 1) * 512]
            patch_mask = mask[:, :, i * 512:(i + 1) * 512]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.squeeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch = patch.repeat(4, 1, 1, 1).cuda()
            with torch.no_grad():
                out = unet(patch, task_id, num_class)
            out = out[0:1]
            out = out[:, :, half_r:half_r + H, :]  # 将padding后的部分切除再比较

            out = out.cpu()
            patch_mask = patch_mask.unsqueeze(0).cpu()

            shp_reg=out.shape
            axes = [0] + list(range(2, len(shp_reg)))
            tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


            TP += tp
            FP += fp
            FN += fn

        if reminder != 0 and i == m:
            patch = img_padding[:, :, W - 512:W]
            patch_mask = mask[:, :, W - reminder:W]

            patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
            patch = patch.squeeze(0)  # 1,3,512,512
            patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

            # DoDnet:
            patch = patch.repeat(4, 1, 1, 1).cuda()
            with torch.no_grad():
                out = unet(patch, task_id, num_class)
            out = out[0:1]
            out = out[:, :, half_r:half_r + H, 512 - reminder:512]  # 将padding后的部分、重合部分切除再比较

            out = out.cpu()
            patch_mask = patch_mask.unsqueeze(0).cpu()

            shp_reg=out.shape
            axes = [0] + list(range(2, len(shp_reg)))
            tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


            TP += tp
            FP += fp
            FN += fn



    f1,iou = calculate_f1_score(TP, FP, FN)
    return f1,iou

def paddingSlideCropF1_3(img, mask,num_class):
    TP = 0
    FP = 0
    FN = 0
    _, H, W = img.shape
    halfPadLen_H = (512 - H) // 2
    halfPadLen_W = (512 - W) // 2

    imgShape=img.shape

    # print("imgShape:",imgShape)

    patch = img
    patch_mask = mask

    patch = padImage(patch, 512, 512)

    patch = torch.Tensor(patch).cuda()  # 将 patch 移动到 GPU 上
    patch = patch.squeeze(0)  # 1,3,512,512
    patch_mask = torch.Tensor(patch_mask).cuda()  # 将 patch_mask 移动到 GPU 上

    # DoDnet:
    #patch = torch.cat([torch.from_numpy(patch)] * 4, dim=1).cuda()  # 将 patch 在通道维度上堆叠四次
    patch = patch.repeat(4, 1, 1, 1).cuda()
    with torch.no_grad():
        out = unet(patch, task_id, num_class)
    out = out[0:1]
    out = out[:, :, halfPadLen_H:halfPadLen_H + H, halfPadLen_W:halfPadLen_W + W]  # 将padding后的部分、重合部分切除再比较

    out = out.cpu()
    patch_mask = patch_mask.unsqueeze(0).cpu()

    shp_reg=out.shape
    axes = [0] + list(range(2, len(shp_reg)))
    tp, fp, fn, _ = get_tp_fp_fn_tn(out, patch_mask,axes)


    TP += tp
    FP += fp
    FN += fn

    f1,iou = calculate_f1_score(TP, FP, FN)
    return f1,iou



model_path = '/home/xuzhengyang/code/universal_segmantation/unet_wimagef_time_2024-05-28_19_58_40_epoch_1050.pkl'
model_state_dict = torch.load(model_path)
weights_dict={}
for k, v in model_state_dict.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

unet.load_state_dict(weights_dict)
unet.eval()

data_folder = '/home/data/xuzhengyang/通用数据集-test'
#TNBC
task_id_dict = {'lizard':0,'TNBC':1,'cpm15':2,'GLAS':3,'cpm17':4,'MonuSeg':5,'Kumar':6,'ConSep':7,'Her2':8,'CRAGv2':9}
channel_dict={'lizard':7,'TNBC':2,'cpm15':2,'GLAS':2,'cpm17':2,'MonuSeg':2,'Kumar':2,'ConSep':8,'Her2':3,'CRAGv2':2}
for dataset in os.listdir(data_folder):
    task_id=task_id_dict[dataset]
    channel=channel_dict[dataset]
    total_F1=torch.zeros(channel)
    total_IOU=torch.zeros(channel)
    img_folder=os.path.join(data_folder,dataset,'img')
    # print(img_folder)
    cnt=0

    for folder1 in os.listdir(img_folder):
        img_path=os.path.join(img_folder,folder1)
        # print(img_path)

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
        F1,IOU = slideCropF1(images, heatmap,num_class)
        total_F1=total_F1+F1
        total_IOU=total_IOU+IOU
        cnt=cnt+1

    mean_F1=total_F1/cnt
    
    mean_IOU=total_IOU/cnt
    total_mean_IOU=torch.mean(mean_IOU[1:])
    print("{} :total_mean_IOU is {}",dataset,total_mean_IOU)
    print("{} :mean IOU score is {}",dataset,mean_IOU)
    
