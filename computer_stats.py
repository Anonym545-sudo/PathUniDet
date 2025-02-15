import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from unet3 import Universal_model

import h5py
from PIL import Image
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from utils import get_inst_centroid

from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
])

metrics = [[], [], [], [], [], []]
paired_all = []  # unique matched index pair


def load_dataset(uni_folder):
    #[]
    #['epithelial','muscle','tumor','connective','inflammatory','nuclei']
    #epithelial：[0:CRAGv2,213:GLAS,378:PanNuke,8279:lizard]
    uni_dataset=OrderedDict()
    for dataset_folder in os.listdir(uni_folder):
        data_path=os.path.join(uni_folder,dataset_folder)

        if not os.path.isdir(data_path):
            continue
        
        #接下来遍历图片数据
        img_path=os.path.join(data_path,'img')
        h5_path=os.path.join(data_path,'ground_truth')

        # print(uni_dataset[dataset_folder])

        if dataset_folder not in uni_dataset:
            # print("cannot find")
            uni_dataset[dataset_folder]=OrderedDict()
            uni_dataset[dataset_folder]['data']=[os.path.join(img_path,img) for img in os.listdir(img_path)]
            uni_dataset[dataset_folder]['gt']=[os.path.join(h5_path,h5) for h5 in os.listdir(h5_path)]
            # random.shuffle(uni_dataset[dataset_folder]['data'])
            # uni_dataset[dataset_folder]['data']=sorted(uni_dataset[dataset_folder]['data'])
        # print(uni_dataset['epithelial']['data'][8279])


    # print(len(uni_dataset['TNBC']['data']))
    return uni_dataset   

def traverse_img(images,heatmap,task_id):
    for i in range(0,H,512):
        for j in range(0,W,512):
            if i+512>=H:bound_i=H
            else:bound_i=i+512
            if j+512>=W:bound_j=W
            else:bound_j=j+512
            image_patch=images[:,i:bound_i,j:bound_j]
            heatmap_patch=heatmap[:,i:bound_i,j:bound_j]
            if image_patch.shape[1] < 512 or image_patch.shape[2] < 512:
                image_patch=padding_img(image_patch,512,512)
            image_patch=image_patch.unsqueeze(0).cuda()
            with torch.no_grad():
                out_patch=unet.forward(image_patch,task_id,num_class)
            # if out_patch.shape[2]!=heatmap_patch
            if out_patch.shape[2]!=heatmap_patch.shape[1] or out_patch.shape[3]!=heatmap_patch.shape[2]:
                out_patch=revert_img(out_patch,heatmap_patch)
            out_patch=out_patch.cpu().detach().numpy()
            out_patch=out_patch.squeeze()
            pred = np.argmax(out_patch, axis=0)
            gt = np.argmax(heatmap_patch, axis=0)
            # run_nuclei_inst_stat(pred, gt)
            run_nuclei_type_stat(pred, gt)

def padding_img(img,width, height):
    # if img.shape[1] < width or img.shape[2] < height:
    pad_width=width-img.shape[1]
    if pad_width%2==0:
        pad_left=pad_width/2
        pad_right=pad_width/2
    else:
        pad_left=pad_width/2
        pad_right=1+pad_width/2

    pad_height=height-img.shape[2]
    if pad_height%2==0:
        pad_up=pad_height/2
        pad_down=pad_height/2
    else:
        pad_up=pad_height/2
        pad_down=1+pad_height/2


    padding=(int(pad_down),int(pad_up),int(pad_left),int(pad_right))
    img=F.pad(img,padding)

    return img

def revert_img(out_batch,heatmap_batch):
    pad_width=out_batch.shape[2]-heatmap_batch.shape[1]
    if pad_width%2==0:
        pad_left=int(pad_width/2)
        pad_right=int(pad_width/2)
        left=int(pad_left)
        right=int(heatmap_batch.shape[1]+pad_right)
    else:
        pad_left=int(pad_width/2)
        pad_right=int(1+pad_width/2)
        left=int(pad_left)
        right=int(heatmap_batch.shape[1]+pad_right)-1

    pad_height=out_batch.shape[3]-heatmap_batch.shape[2]
    if pad_height%2==0:
        pad_up=int(pad_height/2)
        pad_down=int(pad_height/2)
        down=int(pad_down)
        up=int(heatmap_batch.shape[2]+pad_up)
    else:
        pad_up=int(pad_height/2)
        pad_down=int(1+pad_height/2)
        down=int(pad_down)
        up=int(heatmap_batch.shape[2]+pad_up)+1

    out_batch=out_batch[:,:,left:right,down:up]

    return out_batch

def run_nuclei_type_stat(pred, true, type_uid_list=None, exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
                     
    """
    # file_list = glob.glob(pred_dir + "*.mat")
    # file_list.sort()  # ensure same order [1]

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    # for file_idx, filename in enumerate(file_list[:]):
    #     filename = os.path.basename(filename)
    #     basename = filename.split(".")[0]

    #     true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
    # true_centroid = (true_info["inst_centroid"]).astype("float32")
    # true_inst_type = (true_info["inst_type"]).astype("int32")
    true_centroid=get_inst_centroid(true).astype("float32")
    print(true.shape)
    print(true_centroid)
    true_inst_type=true.astype("int32")

    if true_centroid.shape[0] != 0:
        true_inst_type = true_inst_type[:, 0]
    else:  # no instance at all
        true_centroid = np.array([[0, 0]])
        true_inst_type = np.array([0])

    # * for converting the GT type in CoNSeP
    # true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
    # true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4

    # pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
    # # dont squeeze, may be 1 instance exist
    # pred_centroid = (pred_info["inst_centroid"]).astype("float32")
    # pred_inst_type = (pred_info["inst_type"]).astype("int32")
    pred_centroid=get_inst_centroid(pred).astype("float32")
    # print(pred_centroid)
    pred_inst_type=pred.astype("int32")

    if pred_centroid.shape[0] != 0:
        pred_inst_type = pred_inst_type[:, 0]
    else:  # no instance at all
        pred_centroid = np.array([[0, 0]])
        pred_inst_type = np.array([0])

    # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
    paired, unpaired_true, unpaired_pred = pair_coordinates(
        true_centroid, pred_centroid, 12
    )

    # * Aggreate information
    # get the offset as each index represent 1 independent instance
    true_idx_offset = (
        0#true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
    )
    pred_idx_offset = (
        0#pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
    )
    true_inst_type_all.append(true_inst_type)
    pred_inst_type_all.append(pred_inst_type)

    # increment the pairing index statistic
    if paired.shape[0] != 0:  # ! sanity
        paired[:, 0] += true_idx_offset
        paired[:, 1] += pred_idx_offset
        paired_all.append(paired)

    unpaired_true += true_idx_offset
    unpaired_pred += pred_idx_offset
    unpaired_true_all.append(unpaired_true)
    unpaired_pred_all.append(unpaired_pred)


    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(np.array(results_list))
    return


def run_nuclei_inst_stat(pred, true):
    
    true=true.astype("int32")

    
    pred=pred.astype("int32")

    # to ensure that the instance numbering is contiguous
    pred = remap_label(pred, by_size=False)
    true = remap_label(true, by_size=False)

    pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]

    if true.shape[0]>100 and true.shape[1]>100:
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

unet=Universal_model().cuda()
model_path='/home/xuzhengyang/code/universal_segmantation/unet_time_2023-10-30_14_03_48_epoch_2800.pkl'
model_state_dict=torch.load(model_path)
unet.load_state_dict(model_state_dict)
unet.eval()
folder_path='/home/data/xuzhengyang/通用数据集/TNBC/img'
task_id=1
#['lizard':0,'TNBC':1,'cpm15':2,'cpm17':3,'MonuSeg':4,'Kumar':5,'ConSep':6]

for img_root in os.listdir(folder_path):
    img_path=os.path.join(folder_path,img_root)


    h5_path=img_path.replace('img','ground_truth').replace('jpg','h5').replace('tif','h5').replace('png','h5')
    with h5py.File(h5_path, 'r') as hf:
        heatmap=np.array(hf.get('heatmap'))
    num_class,_,_=heatmap.shape
    images = Image.open(img_path)
    images = transforms(images)
    _,H,W=images.shape
    text=np.argmax(heatmap,axis=0)
    print(get_inst_centroid(text))
    traverse_img(images,heatmap,task_id)





print(metrics)
metrics_avg = np.mean(metrics, axis=-1)
print(metrics_avg)
np.set_printoptions(formatter={"float": "{: 0.5f}".format})
print(metrics_avg)
metrics_avg = list(metrics_avg)

            
        


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--mode",
#         help="mode to run the measurement,"
#         "`type` for nuclei instance type classification or"
#         "`instance` for nuclei instance segmentation",
#         nargs="?",
#         default="instance",
#         const="instance",
#     )
#     parser.add_argument(
#         "--pred_dir", help="point to output dir", nargs="?", default="", const=""
#     )
#     parser.add_argument(
#         "--true_dir", help="point to ground truth dir", nargs="?", default="", const=""
#     )
#     args = parser.parse_args()

#     if args.mode == "instance":
#         run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=False)
#     if args.mode == "type":
#         run_nuclei_type_stat(args.pred_dir, args.true_dir)