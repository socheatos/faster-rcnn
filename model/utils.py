import torch 
import numpy as np

# [ ] change np to torch 
# [ ] documentation !
def nms(bbox: torch.Tensor, scores:torch.Tensor, threshold:float):
    '''
    Non maximum suppression algorithm:
        Repeat until there are no predictions left in bbox:
        1. Select the highest score, removing from bbox  
        2. a. Compare the iou between highest score with all predictions 
           b. if iou > threshold for any prediction, remove prediction from bbox

    Parameter
    ---
    :param:`bbox` : ~torch.Tensor 
        [B,N,4] tensor of box predictions of B batch and N anchors
    :param:`scores` : ~torch.Tensor 
        [B,N] tensor of objectness scores of B batch and N anchors
    :param:`threshold` : float
        overlap threshold iou

    Returns
    ---
    :param: `keep` : ~torch.Tensor

    '''
    order = scores.ravel().argsort(descending=True)
    keep = []

    while len(order)>0:
        current_highest_score = order[0]
        keep.append(current_highest_score)
        order = order[1:]

        if len(order)==0:
            break

        ious = iou(bbox[current_highest_score], bbox[order])     
        mask = ious < threshold
        order = order[mask]
    return keep

def boxToLocation(ground_truth, anchors):
    # given bbox, computes offsets and scales to match bbox_1 to bbox_2
    # finding x_a, y_a, h_a, w_a

    h_a = anchors[...,2] - anchors[...,0]
    w_a = anchors[...,3] - anchors[...,1]
    x_a = anchors[...,0] + h_a*0.5
    y_a = anchors[...,1] + w_a*0.5

    # finding x, y, h, w 
    h = ground_truth[...,2] - ground_truth[...,0]
    w = ground_truth[...,3] - ground_truth[...,1]
    x = ground_truth[...,0] + h*0.5
    y = ground_truth[...,1] + w*0.5

    # finding parameterized coordinates of gt associated with the anchors
    # prevents exp overflow
    eps = 1e-10
    h_a += eps
    w_a += eps

    t_y = (y-y_a)/h_a
    t_x = (x-x_a)/w_a 
    t_w = torch.log(w/w_a)
    t_h = torch.log(h/h_a)
    
    t = torch.stack((t_y,t_x,t_h, t_w), dim=-1)

    return t

def locToBox(anchors: torch.Tensor, locs:torch.Tensor):
    # convert anchors to x,y,h,w format
    h_a = anchors[..., 2] - anchors[..., 0]
    w_a = anchors[..., 3] - anchors[..., 1]
    y_a = anchors[..., 0] + 0.5 * h_a
    x_a = anchors[..., 1] + 0.5 * w_a

    # convert locs to x,y,h,w format 
    t_y = locs[...,0::4]     
    t_x = locs[...,1::4]
    t_h = locs[...,2::4]
    t_w = locs[...,3::4]

    y = t_y * h_a.unsqueeze(2) + y_a.unsqueeze(2)
    x = t_x * w_a.unsqueeze(2) + x_a.unsqueeze(2)
    h = torch.exp(t_h) * h_a.unsqueeze(2)
    w = torch.exp(t_w) * w_a.unsqueeze(2)

    rpn_boxes = locs.clone()
    rpn_boxes[...,0::4] = y - (0.5 * h)
    rpn_boxes[...,1::4] = x - (0.5 * w)
    rpn_boxes[...,2::4] = y + (0.5 * h)
    rpn_boxes[...,3::4] = x + (0.5 * w)

    return rpn_boxes

def toXYHW(box):
    '''Converts (Y1,X1,Y2,X2) bounding box to (X,Y,H,W) format where
     (X,Y) are top left coordinates and (H,W) are the heigh and width of the box respectively

    Parameters
    ----------
    box : torch.Tensor
        [B,N,4] tensor 
    
    Returns
    ---
        [B,N,4] torch.Tensor
    '''
    Y1 = box[...,0]
    X1 = box[...,1]
    Y2 = box[...,2]
    X2 = box[...,3]

    H = Y2-Y1
    W = X2-X1

    XYHW = torch.stack((X1,Y1,H,W),dim=-1)
    return XYHW

def toXYXY(box):
    '''Converts (Y1,X1,Y2,X2) bounding box to (X1,Y1,X2,Y2) format where

    Parameters
    ----------
    box : torch.Tensor
        [B,N,4] tensor 
    
    Returns
    ---
        [B,N,4] torch.Tensor
    '''
    Y1 = box[...,0]
    X1 = box[...,1]
    Y2 = box[...,2]
    X2 = box[...,3]

    XYXY = torch.stack((X1,Y1,X2,Y2),dim=-1)
    return XYXY

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

def iou_batch(groundtruth,anchors):
    '''
    Batch calculation o
    
    '''
    anchors_cut_area = (anchors[...,2]-anchors[...,0]) * (anchors[...,3]-anchors[...,1])
    gt_area = (groundtruth[...,2]-groundtruth[...,0]) * (groundtruth[...,3]-groundtruth[...,1])

    maxx = torch.max(anchors.unsqueeze(2).permute(1,0,2,3)[...,:2], groundtruth[...,:2]).permute(1,2,0,3)
    minn = torch.min(anchors.unsqueeze(2).permute(1,0,2,3)[...,2:], groundtruth[...,2:]).permute(1,2,0,3)

    intersect = torch.clip(minn[...,0]-maxx[...,0],0) * torch.clip(minn[...,1]-maxx[...,1],0)
    iou = (intersect/((anchors_cut_area.unsqueeze(1)+ gt_area.unsqueeze(-1))-intersect)).permute(0,2,1)
    return iou

def sampling(labels: torch.Tensor, n_sample: int, pos_ratio: float=0.5):
    '''
    Randomly samples from labels with given number of sample and ratio of postive/negative.
    First sampling positive labels, if it does not meet the threshold, the sample will be padded with negative labels.
    
    Parameter:
    ---
    :parameter:`labels` `~torch.Tensor` (BxN) where B is the batch size and N is the number of labels
    :parameter:`n_sample` :class:`int`
    :parameter:`pos_ratio` :class:`float`

    Return: 
        `~torch.Tensor`
    '''

    n_pos = int(n_sample*pos_ratio)
    pos_idx = (torch.where(labels==1)[0]).float()

    if len(pos_idx) > n_pos:
        discard = pos_idx[pos_idx.multinomial(len(pos_idx)-n_pos, replacement=False)].long()
        labels[discard] = -1

    pos_idx = torch.where(labels==1)[0]
    n_neg = n_sample - len(pos_idx)
    neg_idx = (torch.where(labels==0)[0]).float()

    if len(neg_idx) > n_neg:
        discard = neg_idx[neg_idx.multinomial(len(neg_idx)-n_neg, replacement=False)].long()
        labels[discard] = -1
    
    return labels
  

def assignLabels(label, ious, pos_iou_thres=0.5):
    '''
    Parameters
    ----------
    label : torch.Tensor
        [batch_size, num_obs] of -1 filled tensors
    ious : torch.Tensor
        [batch_size, num_obs, num_labels] 
    pos_ious_thres : float, optional
        label=1 if above the threshold and 0 if below 1-threshod, by default 0.5
    '''
    label_max_ious, _ = torch.max(ious, dim=1)
    label_max_ious[torch.where(label_max_ious==0)] = -1
    all_label_max_idx = torch.where(ious==label_max_ious.unsqueeze(1))
    
    _, gt_idx =  torch.max(ious, dim=-1)
    gt_ious = torch.gather(ious,-1,gt_idx.unsqueeze(-1)).squeeze(-1)
    gt_max_ious, _ = torch.max(gt_ious, dim=1)
    gt_max_idx = torch.where(gt_ious==gt_max_ious.unsqueeze(-1))

    label[gt_ious<(1-pos_iou_thres)] = 0
    label[(all_label_max_idx[0],all_label_max_idx[1])] = 1
    label[gt_max_idx] = 1
    label[gt_ious>=pos_iou_thres] = 1

    return label, gt_idx

def generateBBox(gt_boxes, idx):
    gt_idx_anchors = idx.repeat(4,1,1).permute(1,2,0).unsqueeze(-2)
    gt_boxes = gt_boxes.unsqueeze(1).repeat(1,idx.size(1),1,1)
    gt_boxes = torch.gather(gt_boxes, -2, gt_idx_anchors).squeeze()
    return gt_boxes

def generateClsLab(fg_label, obj_label,idx):

    cls_label = torch.empty_like(fg_label)
    cls_label.fill_(-1)

    gt_idx_objlab = idx.unsqueeze(1)
    obj_label = obj_label.repeat(1,1,idx.size(1)).squeeze()
    selected = torch.gather(obj_label,1,gt_idx_objlab).squeeze().type_as(cls_label) #(BATCH x NUMANCHORS)

    mask = fg_label.eq(1)

    cls_label[mask] = selected[mask]
    cls_label +=1

    return cls_label