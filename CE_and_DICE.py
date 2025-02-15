import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Focal_loss import FocalLoss

def cross_entropy_loss(regression, class_heat_maps):
    class_heat_maps = class_heat_maps.type(torch.float32)
    
    regression_loss=-class_heat_maps*torch.log(torch.clip(regression,1e-10,1.0))
    # print(regression_loss.shape)
    regression_loss = torch.sum(regression_loss, dim=1)
    # print(regression_loss.shape)
    regression_loss = torch.mean(regression_loss)
   
    return regression_loss

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

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


    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2



    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

def dice_loss(regression, class_heat_maps,smooth=1.0):
    shp_reg=regression.shape
    axes = [0] + list(range(2, len(shp_reg)))
    tp, fp, fn, _ = get_tp_fp_fn_tn(regression, class_heat_maps, axes)
    nominator = 2 * tp + smooth
    denominator = 2 * tp + fp + fn + smooth

    dc = nominator / (denominator + 1e-8)

    dc = dc.mean()

    return 1-dc


def CE_loss(regression, class_heat_maps,lam=1.0):
    cross_entropy=2*cross_entropy_loss(regression, class_heat_maps)

    return cross_entropy,cross_entropy,cross_entropy

def cross_and_dice_loss(regression, class_heat_maps,lam=1.0):
    cross_entropy=cross_entropy_loss(regression, class_heat_maps)
    dice=dice_loss(regression, class_heat_maps)

    loss=cross_entropy+lam*dice

    
    return loss,cross_entropy,dice


def L1_loss(regression, class_heat_maps):
    return F.l1_loss(regression, class_heat_maps, reduction='mean')




# from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         # target = target.view(-1,1)
#         target = target.view(-1, 1).type(torch.long)  # Ensure target is of type int64 (torch.long)
#         print(input.shape)
#         print(target.shape)
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()



