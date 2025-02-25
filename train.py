import argparse

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


from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from Universal import DLASeg
from cross_entropy import cross_entropy_loss
from CE_and_DICE import cross_and_dice_loss,CE_loss
from bounding_losses import FocalLoss,RegL1Loss

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg

import gc
import h5py


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
])

def get_arguments():
    parser = argparse.ArgumentParser(description="PathUniDet")

    parser.add_argument("--dataset_dir", type=str, default='/media/ipmi2022/SCSI_all/universal_data', help="your training dataset path for all tasks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--dataset_num", type=int, default=11)
    parser.add_argument("--sampling", type=int, default=30,help="sampling images per epoch for each dataset")
    parser.add_argument("--model_path", type=str, help="the pretrained path, you can download in the link")
    parser.add_argument("--lr", type=int, default=0.0001)
    parser.add_argument(
    "--tasks",
    type=str,
    nargs='+', 
    default=[
        'task01_camelyon', 'task02_Her2_region', 'task03_Gland',
        'task04_PanNuke', 'task05_ConSep', 'task06_lizard',
        'task07_HP', 'task08_Ki67', 'task09_MBM', 'task10_Her2', 'task11_nuclei'
    ],
    )
    parser.add_argument(
        "--seg_num",
        type=str,
        default="(0,1,2,3,4,5,9,10)",
        help="the task number for segmentation tasks"
    )
    parser.add_argument(
        "--det_num",
        type=str,
        default="(6,7,8)",
        help="the task number for detection tasks"
    )


    return parser


def load_dataset(uni_folder):
    
    uni_dataset=OrderedDict()
    task=args.tasks

    for num in range(len(task)):
        data_path=os.path.join(uni_folder,task[num])

        if not os.path.isdir(data_path):
            continue
        
        img_path=os.path.join(data_path,'img')
        h5_path=os.path.join(data_path,'ground_truth')


        if task[num] not in uni_dataset:
            uni_dataset[task[num]]=OrderedDict()
            uni_dataset[task[num]]['data']=[os.path.join(img_path,img) for img in os.listdir(img_path)]
            uni_dataset[task[num]]['gt']=[os.path.join(h5_path,h5) for h5 in os.listdir(h5_path)]
            random.shuffle(uni_dataset[task[num]]['data'])


    return uni_dataset      

def randomCrop(img, mask, width, height):
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]

    if img.shape[1] >= width and img.shape[2] >= height:
        x = random.randint(0, img.shape[1] - height)
        y = random.randint(0, img.shape[2] - width)
        img = img[:,x:x+width,y:y+height]
        mask = mask[:,x:x+width,y:y+height]

    elif img.shape[1] >= width and img.shape[2] < height:
        x = random.randint(0, img.shape[1] - height)
        # y = random.randint(0, img.shape[2] - width)
        img = img[:,x:x+width,:]
        mask = mask[:,x:x+width,:]

    elif img.shape[1] < width and img.shape[2] >= height:
        # x = random.randint(0, img.shape[1] - height)
        y = random.randint(0, img.shape[2] - width)
        img = img[:,:,y:y+height]
        mask = mask[:,:,y:y+height]

    
    if img.shape[1] < width or img.shape[2] < height:
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


    
    return img, mask

def split_dataset(dataset,train_partial=80):

    train_dataset=OrderedDict()
    test_dataset=OrderedDict()
    train_dataset_len=[]
    print(dataset.keys)

    for dataset_name in dataset.keys():
        print(dataset_name)
        dataset_len=len(dataset[dataset_name]['data'])
        train_dataset_len.append(int(dataset_len*1.0))
        print(dataset_len)
        train_dataset[dataset_name]={}
        test_dataset[dataset_name]={}

        train_dataset[dataset_name]['data']=dataset[dataset_name]['data'][0:int(dataset_len*1.0)]
        train_dataset[dataset_name]['gt']=dataset[dataset_name]['gt'][0:int(dataset_len*1.0)]
        test_dataset[dataset_name]['data']=dataset[dataset_name]['data'][int(dataset_len*1.0):dataset_len]
        test_dataset[dataset_name]['gt']=dataset[dataset_name]['gt'][int(dataset_len*1.0):dataset_len]
          
      
    return train_dataset,test_dataset,train_dataset_len


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None,task_num=11,train_dataset_len=[],transforms=None):
        super().__init__(data, batch_size, number_of_threads_in_multithreaded)
        self.batch_size=batch_size
        self.dataset=data
        self.list_of_keys=args.tasks
        self.task_id=[i for i in range(len(data.keys()))]


        # self.task_pool=30*np.ones(len(data.keys()),dtype=int)
        self.task_pool=30*np.ones(len(data.keys()),dtype=int)
        self.train_dataset_len=train_dataset_len

        self.transforms=transforms

    def __next__(self):
        return self.generate_train_batch(self)

    def generate_train_batch(self,number_of_threads_in_multithreaded=None):
        selected_keys = np.random.choice(self.task_id, 1, True, None)
        
        
        selected_keys=selected_keys.item()

        self.task_pool[selected_keys]-=1
        while self.task_pool[selected_keys]==0:
            if all(task==0 for task in self.task_pool):
                self.task_pool=30*np.ones(len(self.dataset.keys()),dtype=int)
            selected_keys = np.random.choice(self.task_id, 1, True, None)
            
            selected_keys=selected_keys.item()
                
        dataset_name=self.list_of_keys[selected_keys]
        

        select_data_index=np.random.choice(self.train_dataset_len[selected_keys],self.batch_size,False,None)
        h5_root=self.dataset[dataset_name]['gt'][0]

        with h5py.File(h5_root, 'r') as hf:
            class_heat_map=np.array(hf.get('heatmap'))
        num_class,_,_=class_heat_map.shape

        data=torch.zeros(self.batch_size,3,patch_size,patch_size)
        heatmap=[]

        for batch,index in enumerate(select_data_index):

            img_root=self.dataset[dataset_name]['data'][index]

            _,split_name=os.path.split(img_root)              
            while split_name=='.DS_Store':
                index=np.random.choice(self.train_dataset_len[selected_keys],1,False,None)
                img_root=self.dataset[dataset_name]['data'][index.item()]
                _,split_name=os.path.split(img_root)

            h5_root=img_root.replace('img','ground_truth').replace('.jpg','.h5').replace('.png','.h5').replace('.tif','.h5')

            with h5py.File(h5_root, 'r') as hf:
                class_heat_map=np.array(hf.get('heatmap'))

            image = Image.open(img_root)
            if self.transforms is not None:
                image = self.transforms(image)


            image, class_heat_map=randomCrop(image, class_heat_map, width=patch_size, height=patch_size)
          
            class_heat_map=torch.tensor(class_heat_map)
            data[batch]=image
            heatmap.append(class_heat_map)

        return data,heatmap,num_class,selected_keys


def main():

    parser = get_arguments()
    print(parser)

    args = parser.parse_args()

    patch_size=512
    uni_dataset=load_dataset('/media/ipmi2022/SCSI_all/xuzhengyang/universal_data')
    train_uni_dataset,test_uni_dataset,train_dataset_len=split_dataset(uni_dataset)
    data_generator=DataLoader(train_uni_dataset,batch_size=args.batch_size,train_dataset_len=train_dataset_len,transforms=transforms)
    
    heads={'seg':15,'kp':4 ,'hm': 4, 'wh': 2, 'reg': 2}
    unet=DLASeg('dla34', heads,
                     pretrained=True,
                     down_ratio=2,
                     head_conv=256).cuda()
    
    unet.train()
    model_state_dict=torch.load(args.model_path)

    current_state_dict = unet.state_dict()
    
    for name, param in model_state_dict.items():
        if name in current_state_dict and current_state_dict[name].shape == param.shape:
            current_state_dict[name] = param
        else:
            print(f"Skipping {name} due to shape mismatch.")
    
    unet.load_state_dict(current_state_dict)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    
    
    epochs=args.epochs
    for epoch in tqdm(range(1,epochs),desc='Epoch'):
        mean_loss=0
        mean_ce=0
        mean_dice=0
       
        for it in tqdm(range(args.sampling*args.dataset_num)):
            data,heatmap,num_class,selected_keys = next(data_generator)
            optimizer.zero_grad()
    
            data=data.cuda()
            B,_,_,_=data.shape
                    
            
            out=unet.forward(data,selected_keys)
            out=out[:,:num_class]
            out=F.softmax(out,dim=1)
            loss=0
            cross=0
            dice=0
            
            for batch in range(B):
                out_batch=out[batch:batch+1]
                heatmap_batch=heatmap[batch].reshape(1,num_class,heatmap[batch].shape[1],heatmap[batch].shape[2])
                heatmap_batch=heatmap_batch.cuda()
    
    
                if out_batch.shape[2]==heatmap_batch.shape[2] and out_batch.shape[3]==heatmap_batch.shape[3]:
    
                    if selected_keys in args.seg_num:
                        batch_loss,batch_cross,batch_dice=cross_and_dice_loss(out_batch,heatmap_batch)
                    elif selected_keys in args.det_num:
                        batch_loss,batch_cross,batch_dice=CE_loss(out_batch,heatmap_batch)
                    loss=loss+batch_loss
                    cross=cross+batch_cross
                    dice=dice+batch_dice
    
                else:
                    pad_width=out_batch.shape[2]-heatmap_batch.shape[2]
                    if pad_width%2==0:
                        pad_left=int(pad_width/2)
                        pad_right=int(pad_width/2)
                        left=int(pad_left)
                        right=int(heatmap_batch.shape[2]+pad_right)
                    else:
                        pad_left=int(pad_width/2)
                        pad_right=int(1+pad_width/2)
                        left=int(pad_left)
                        right=int(heatmap_batch.shape[2]+pad_right)-1
    
                    pad_height=out_batch.shape[3]-heatmap_batch.shape[3]
                    if pad_height%2==0:
                        pad_up=int(pad_height/2)
                        pad_down=int(pad_height/2)
                        down=int(pad_down)
                        up=int(heatmap_batch.shape[3]+pad_up)
                    else:
                        pad_up=int(pad_height/2)
                        pad_down=int(1+pad_height/2)
                        down=int(pad_down)
                        up=int(heatmap_batch.shape[3]+pad_up)+1
    
    
                    out_batch=out_batch[:,:,left:right,down:up]
                    heatmap_batch=heatmap_batch.type(torch.int64)
    
                    if selected_keys in args.seg_num:
                        batch_loss,batch_cross,batch_dice=cross_and_dice_loss(out_batch,heatmap_batch)
                    elif selected_keys in args.det_num:
                        batch_loss,batch_cross,batch_dice=CE_loss(out_batch,heatmap_batch)
                    loss=loss+batch_loss
                    cross=cross+batch_cross
                    dice=dice+batch_dice
    
    
    
            loss=loss/B
            cross=cross/B
            dice=dice/B
            loss.backward()
            optimizer.step()
            mean_loss=mean_loss+loss
            mean_ce=mean_ce+cross
            mean_dice=mean_dice+dice
            
            del data, heatmap, num_class, selected_keys
            gc.collect()
        mean_loss=mean_loss/args.sampling*args.dataset_num
        mean_ce=mean_ce/args.sampling*args.dataset_num
        mean_dice=mean_dice/args.sampling*args.dataset_num
    
        tqdm.write(f"Epoch {epoch}: Loss={mean_loss:.4f}, cross entropy={mean_ce:.4f}, dice={mean_dice:.4f}")
        if epoch%50==0 and epoch!=0:
            now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            save_name=f'PathUniDet_time_newdata_{now}_epoch_{epoch}.pkl'
            torch.save(unet.state_dict(), save_name)
    
    
    
    now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    save_name=f'PathUniDet_time_newdata_{now}_final.pkl'
    torch.save(unet.state_dict(), save_name)
