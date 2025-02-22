import os
import numpy as np
import cv2
import h5py
from skimage import draw
import xml.etree.ElementTree as ET
from tqdm import tqdm
import math

for h in range(1,38):
    data_path = 'D:/MoNuSeg 2018 Training Data/Annotations/'+str(h)+'.xmL'
    tree = ET.parse(data_path)
    root = tree.getroot()
    child = root[0]

    cell_mask = np.zeros((1000, 1000))
    back_mask = np.ones((1000, 1000))

    for x in child: 
        r = x.tag
        if r == 'Regions':
            for y in x:
                y_tag = y.tag

                if y_tag == 'Region':

                    regions = []
                    vertices = y[1]
                    coords = np.zeros((len(vertices), 2))                       
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = math.floor(float(vertex.attrib['X']))
                        coords[i][1] = math.floor(float(vertex.attrib['Y']))
                    regions.append(coords)

                    vertex_row_coords = regions[0][:,0]
                    vertex_col_coords = regions[0][:,1]
                    points = np.zeros((len(vertices), 2))
                    fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, cell_mask.shape)
                    cell_mask[fill_row_coords, fill_col_coords] = 1
                    back_mask[fill_row_coords, fill_col_coords] = 0

    heatmap = [back_mask, cell_mask]

    with h5py.File(str(h+14)+".h5", 'a') as hf:
        Xset = hf.create_dataset(name='heatmap', data=heatmap)

