import torch 
import numpy as np

def nms(bbox: torch.Tensor, scores:torch.Tensor, threshold:float):
    '''
    Parameter
    ---
    :param:`bbox` ~torch.Tensor [N,4]
        2-D tensor of box predictions
    :param:`scores` ~torch.Tensor [N,1]
        1-D tensor of objectness scores
    :param:`threshold`  ~float
        overlap threshold iou
    '''
    order = scores.ravel().argsort(descending=True)
    keep = []

    while len(order)>0:
    # 1.    select prediction S with highest confidence score add it to KEEP and remove from bbox
        current_highest_score = order[0]
        keep.append(current_highest_score)
        order = order[1:]

        # sanity check
        if len(order)==0:
            break

    # 2.a   compare prediction S with all predictions present in bbox by comparing the iou
        ious = iou(bbox[current_highest_score], bbox[order])     
    # 2.b   if iou > threshold for any prediction b in bbox, remove b from bbox
        mask = ious < threshold
        order = order[mask]
    # 3.    repeat until no predictions left in bbox 
    return keep

def boxToLocation(ground_truth, anchors):
    # given bbox, computes offsets and scales to match bbox_1 to bbox_2
    # finding x_a, y_a, h_a, w_a
    h_a = anchors[:,:,2] - anchors[:,:,0]
    w_a = anchors[:,:,3] - anchors[:,:,1]
    x_a = anchors[:,:,0] + h_a*0.5
    y_a = anchors[:,:,1] + w_a*0.5

    # finding x, y, h, w 
    h = ground_truth[:,:,2] - ground_truth[:,:,0]
    w = ground_truth[:,:,3] - ground_truth[:,:,1]
    x = ground_truth[:,:,0] + h*0.5
    y = ground_truth[:,:,1] + w*0.5

    # finding parameterized coordinates of gt associated with the anchors
    # prevents exp overflow
    eps = np.finfo(h_a.dtype).eps
    h_a = np.maximum(h_a, eps)
    w_a = np.maximum(w_a, eps)

    t_y = (y-y_a)/h_a
    t_x = (x-x_a)/w_a 
    t_w = np.log(w/w_a)
    t_h = np.log(h/h_a)

    return t_y,t_x,t_w,t_h

def locToBox(anchors: torch.Tensor, locs:torch.Tensor):
    # convert anchors to x,y,h,w format
    h_a = anchors[:,:, 2] - anchors[:,:, 0]
    w_a = anchors[:,:, 3] - anchors[:,:, 1]
    y_a = anchors[:,:, 0] + 0.5 * h_a
    x_a = anchors[:,:, 1] + 0.5 * w_a

    # convert locs to x,y,h,w format 
    t_y = locs[:,:,0::4]     
    t_x = locs[:,:,1::4]
    t_h = locs[:,:,2::4]
    t_w = locs[:,:,3::4]

    y = t_y * h_a.unsqueeze(2) + y_a.unsqueeze(2)
    x = t_x * w_a.unsqueeze(2) + x_a.unsqueeze(2)
    h = torch.exp(t_h) * h_a.unsqueeze(2)
    w = torch.exp(t_w) * w_a.unsqueeze(2)

    rpn_boxes = locs.clone()
    rpn_boxes[:,:,0::4] = y - (0.5 * h)
    rpn_boxes[:,:,1::4] = x - (0.5 * w)
    rpn_boxes[:,:,2::4] = y + (0.5 * h)
    rpn_boxes[:,:,3::4] = x + (0.5 * w)

    return rpn_boxes

def generateCenters(feat_stride, height, width):
    cntr_x = np.arange(feat_stride, (width+1)*feat_stride, feat_stride) - width/2
    cntr_y = np.arange(feat_stride, (height+1)*feat_stride, feat_stride) - height/2
    centers = np.array(np.meshgrid(cntr_x,cntr_y)).T.reshape(-1,2)
    return centers

def generateAnchors(ratios, anchor_scales,feat_stride, height, width):
    # for each center, generate len(ratios)*len(anchor_scales) there
    # there are centers_row*centers_col*len(ratios)*len(anchor_scales) in total
    # for each center, generate the anchor boxes 
    centers = generateCenters(feat_stride, height, width)

    anchors = np.zeros((len(centers)*len(anchor_scales)*len(ratios),4))
    idx = 0 
    for cntr_y, cntr_x in centers:
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = feat_stride * anchor_scales[j] * np.sqrt(ratios[i])
                w = feat_stride * anchor_scales[j] * np.sqrt(1/ratios[i])

                anchors[idx, 0] = cntr_y - h /2
                anchors[idx, 1] = cntr_x - w /2
                anchors[idx, 2] = cntr_y + h /2
                anchors[idx, 3] = cntr_x + w /2
                
                idx +=1

    # return anchors.astype(np.int32)
    return anchors


def split(batch, keep):
    counts = torch.unique(keep[0],return_counts=True)[1]
    counts = torch.Tensor.cpu(counts)
    splits = tuple(counts.numpy())
    try:
        batch = batch[keep[0],keep[1],...]
    except:
        batch = batch[keep[0],keep[1]]

    batch = torch.split(batch, split_size_or_sections=splits)
    return batch

def iou(groundtruth, anchor):
    y1,x1,y2,x2 = anchor[...,0],anchor[...,1],anchor[...,2],anchor[...,3]
    gy1,gx1,gy2,gx2 = groundtruth[...,0],groundtruth[...,1],groundtruth[...,2],groundtruth[...,3]

    xA = torch.max(x1,gx1)
    yA = torch.max(y1,gy1)
    xB = torch.min(x2,gx2)
    yB = torch.min(y2,gy2)

    ia = torch.clip(xB-xA,0) * torch.clip(yB-yA,0)

    boxAarea =  (x2-x1)*(y2-y1)
    boxBarea = (gx2-gx1)*(gy2-gy1)
    iou = ia/(boxAarea+boxBarea-ia)
    return iou


def sampling(labels: torch.Tensor, n_sample: int, pos_ratio: float=0.5):
    '''
    Randomly samples from labels with given number of sample and ratio of postive/negative
    '''

    n_pos = int(n_sample*pos_ratio)
    pos_idx = (torch.where(labels==1)[0]).float()
    # print(pos_idx)
    print(len(pos_idx))
    if len(pos_idx) > n_pos:
        discard = pos_idx[pos_idx.multinomial(len(pos_idx)-n_pos, replacement=False)].long()
        labels[discard] = -1

    pos_idx = torch.where(labels==1)[0]
    n_neg = n_sample - len(pos_idx)
    neg_idx = (torch.where(labels==0)[0]).float()

    if len(neg_idx) > n_neg:
        discard = neg_idx[neg_idx.multinomial(len(neg_idx)-n_neg, replacement=False)].long()
        labels[discard] = -1

    print(len(pos_idx),n_neg)
    # print(len(torch.where(labels==1)[0]))
    return labels
  


