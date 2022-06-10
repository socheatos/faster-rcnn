import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    return tuple(zip(*batch))

# def pad_labels(batch):
#     '''Make all tensor in batch the same length by padding with zeros'''
#     batch = [i]


def collate_fn_2(batch):
    images = list()
    labels = list()
    bboxs = list()
    for b in batch:
        images.append(b[0])
        labels.append(b[1])
        bboxs.append(b[2])
    images = torch.stack(images, dim=0)
    # labels = pad_sequence(labels, batch_first=True)
    # bboxs = pad_sequence(bboxs, batch_first=True)

    return images, labels, bboxs

