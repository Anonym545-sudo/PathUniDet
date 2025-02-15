import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cross_entropy_loss(regression, class_heat_maps):


    class_heat_maps = class_heat_maps.type(torch.float32)


    regression_loss=-class_heat_maps*torch.log(torch.clip(regression,1e-10,1.0))
    # print(regression_loss.shape)
    regression_loss = torch.sum(regression_loss, dim=1)
    # print(regression_loss.shape)
    regression_loss = torch.mean(regression_loss)
   
    return regression_loss