import enum
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import cv2 

def collate_fn(batch):
    images = list()
    labels = list()
    bboxs = list()
    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxs.append(b[2])
    images = torch.stack(images, dim=0)
    labels = pad_sequence(labels, batch_first=True)
    bboxs = pad_sequence(bboxs, batch_first=True)

    return images, labels, bboxs

def visualize_images(bbox_proposed, images, bboxs):
    plt.figure(figsize=(60,30))
    for i, im in enumerate(images):
        image_ = im.cpu().squeeze(0).permute(1,2,0)
        image_ = cv2.cvtColor(np.copy(image_), cv2.COLOR_BGR2RGB)
        bbox_ = bboxs[i] #ground truth
        prop_bbox = bbox_proposed[i]

        for j in range(len(bbox_)):
            image_ = cv2.rectangle(image_, (int(bbox_[j][1]),int(bbox_[j][0])),(int(bbox_[j][3]),int(bbox_[j][2])),color=(0,255,0),thickness=1)
                       
        for k in prop_bbox:
            image_ = cv2.rectangle(image_, (int(k[1]),int(k[0])),(int(k[3]),int(k[2])),color=(0,0,255),thickness=2)
        
        plt.subplot(1,len(images),i+1)
        plt.imshow(image_)
        plt.axis('off')
    
    plt.show()