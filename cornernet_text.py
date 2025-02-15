from Corner.CornerNet import pool,CornerNetwork,model
from Corner.py_utils.kp_utils import _tranpose_and_gather_feat, _decode


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


from datetime import datetime


import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', message='.*masked_fill_ received a mask with dtype torch.uint8.*')
# warnings.filterwarnings('ignore', message='.*indexing with dtype torch.uint8 is now deprecated.*')
# warnings.filterwarnings('ignore', message='.*An output with one or more elements was resized since it had shape.*')
# warnings.filterwarnings('ignore', message='.*masked_scatter_ received a mask with dtype torch.uint8.*')


# h5_root='/home/xingzehang/project_16t/xuzhengyang/TCT/train_data/cornernet_ground_truth/0.h5'
# img_root='/home/xingzehang/project_16t/xuzhengyang/TCT/train_data/images/0.png'

import sys
sys.path.append('/home/xuzhengyang/code/universal_segmantation/Corner')
from Corner.py_utils import kp, AELoss, _neg_loss, convolution, residual 


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
])


class Cell_Dataset:
    def __init__(self, data_root, gt_root, transform=None):
            self.root1 = data_root
            self.root2 = gt_root
            self.transform = transform
            self.image_files = sorted(os.listdir(data_root))
            self.h5_files = sorted(os.listdir(gt_root))


    def __getitem__(self, index):
        h5_path = os.path.join(self.root2, self.h5_files[index])
        img_path=h5_path.replace('cornernet_ground_truth','img').replace('.h5','.png')
        print(h5_path)

        if not os.path.exists(img_path):
             img_path=img_path.replace('.JPG','.png')

        with h5py.File(h5_path, 'r') as hf:
            tl_heatmap = np.array(hf.get('tl_heatmap'))
            br_heatmap=np.array(hf.get('br_heatmap'))
            tl_regrs=np.array(hf.get('tl_regrs'))
            br_regrs=np.array(hf.get('br_regrs'))
            tl_tags=np.array(hf.get('tl_tags'))
            br_tags=np.array(hf.get('br_tags'))
            tag_masks=np.array(hf.get('tag_masks'))
            
        image = Image.open(img_path)
        image = transforms(image).cuda()

        return image,tl_heatmap,br_heatmap,tl_regrs,br_regrs,tl_tags,br_tags,tag_masks
    
    def __len__(self):
        return len(self.h5_files)
# def reshape_tensor(matrix):
    
train_dataset = Cell_Dataset(data_root='/home/xuzhengyang/project_16t/xuzhengyang/TCT/train_data/img', gt_root='/home/xuzhengyang/project_16t/xuzhengyang/TCT/train_data/cornernet_ground_truth', transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# print(image.shape)
# Pool=CornerNetwork().cuda()
Pool=model().cuda()
Pool.train()



# model_path='/home/xuzhengyang/code/universal_segmantation/corner_epoch_5_time_2024-06-26_02_50_05.pkl'
# model_state_dict=torch.load(model_path)
# Pool.load_state_dict(model_state_dict)
optimizer = torch.optim.Adam(Pool.parameters(), lr=0.0001)#,weight_decay=0.005)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)
ae_loss=AELoss()


epochs=1000
for epoch in tqdm(range(4,epochs),desc='Epoch'):
    epoch_loss=0
    for image,tl_heatmap,br_heatmap,tl_regrs,br_regrs,tl_tags,br_tags,tag_masks in tqdm(train_dataloader):
        x=(image,tl_tags,br_tags)

        out=Pool(*x)

        # tl_tags=torch.tensor(tl_tags).cuda()
        # br_tags=torch.tensor(br_tags).cuda()

        # # 在输出特征图上，取物体的gt bbox的角点对应位置的值（可以是embedding，也可以是regr）
        # tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_tags)
        # br_tag  = _tranpose_and_gather_feat(br_tag, br_tags)
        # tl_regr = _tranpose_and_gather_feat(tl_regr, tl_tags)
        # br_regr = _tranpose_and_gather_feat(br_regr, br_tags)

        # # print(tl_tag.shape)
        # out=[tl_heat, br_heat, tl_tag,  br_tag, tl_regr, br_regr]
        target=[tl_heatmap,br_heatmap,tag_masks,tl_regrs,br_regrs]
        target = [torch.tensor(item).cuda() for item in target]
        # target = [target]

        loss=ae_loss(out,target)
        print(loss)
        # print(out.shape)
        # print(target.shape)

        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss=epoch_loss+loss

    epoch_loss=epoch_loss/len(train_dataloader)

    tqdm.write(f"Epoch {epoch}: Loss={epoch_loss.item():.4f}")

    if epoch%10==0 and epoch >= 50:
        now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        save_name=f'corner_epoch_{epoch}_time_{now}.pkl'
        torch.save(Pool.state_dict(), save_name)


now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
save_name=f'corner_final_{epoch}_time_{now}.pkl'
torch.save(Pool.state_dict(), save_name)
# print(out)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)
# print(out[3].shape)
# print(out[4].shape)
# print(out[5].shape)


# print(out[0::6].shape)
# print(tl_tag.shape)
# tl_tag=tl_regr.reshape(1,128,128,2)
# def _tranpose_and_gather_feat(feat, ind):
#     feat = feat.permute(0, 2, 3, 1).contiguous()
#     # print(feat.shape)
#     feat = feat.view(feat.size(0), -1, feat.size(3))
#     # print(feat.shape)
#     # feat = _gather_feat(feat, ind)
#     return feat

# tl_tag=_tranpose_and_gather_feat(tl_tag,1)

# print(tl_tag.shape)