import numpy as np
import h5py
import os
import cv2

data_root='/Users/xuzhengyang/Downloads/archive/Annotator 1 (biologist)/mask binary'
save_root='/Users/xuzhengyang/Downloads/archive/ground_truth'
for np_file in os.listdir(data_root):
    h5_path=os.path.join(data_root,np_file)
    gt=cv2.imread(h5_path, cv2.IMREAD_GRAYSCALE)
    heatmap=np.zeros((2,gt.shape[0],gt.shape[1]))
    heatmap[0]=(gt==0)
    heatmap[1]=(gt==255)

   
    store_name=os.path.join(save_root,np_file).replace('.png','.h5')

    with h5py.File(store_name, 'w') as f:
        f.create_dataset('heatmap', data=heatmap)

