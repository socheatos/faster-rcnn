from numpy import dtype
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import os


class ImageDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None, target_transorm=None):
        self.img_dir = img_dir
        self.anotations = pd.read_csv(annotations,names=['image_path','label','bbox'],index_col=False,sep=';',converters={'bbox':pd.eval,'label':pd.eval})
        self.transform = transform
        self.target_transorm = target_transorm

        self.bboxs = self.anotations.bbox.to_numpy()
        self.labels = self.anotations.label.to_numpy()

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, index):
        img_pth = os.path.join(self.img_dir,self.anotations.iloc[index,0])
        image = read_image(img_pth)
        _,im_h, im_w = image.shape
        # print(im_h,im_w)
        label = torch.tensor(self.labels[index].astype('int'))
        bbox = torch.tensor(self.bboxs[index].astype('float'))

        if self.transform:
            image = self.transform(image)
            _, new_h, new_w = image.shape
            scale_h, scale_w = new_h/im_h, new_w/im_w
            bbox[...,0::2] = bbox[...,0::2]*scale_h
            bbox[...,1::2] = bbox[...,1::2]*scale_w
            
        
        return image, label, bbox
        



        