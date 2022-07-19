import torch 

# [x] change np to torch 
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
        keep.append(current_highest_score.item())
        order = order[1:]

        if len(order)==0:
            break

        ious = iou(bbox[current_highest_score], bbox[order])     
        mask = ious < threshold
        order = order[mask]
    return keep

def boxToLoc(ground_truth, anchors):
    # given bbox, computes offsets and scales to match bbox_1 to bbox_2
    # finding x_a, y_a, h_a, w_a
    eps = torch.tensor(1e-6)

    if ground_truth.size(-1) == 5:
        ground_truth = ground_truth[...,1:]
    if anchors.size(-1)==5:
        anchors = anchors[...,1:]
    if ground_truth.dim() != anchors.dim():
        anchors = anchors.unsqueeze(0).repeat(ground_truth.size(0),1,1)

    h_a = torch.max(anchors[...,2] - anchors[...,0],eps)
    w_a = torch.max(anchors[...,3] - anchors[...,1], eps)
    x_a = anchors[...,0] + h_a*0.5
    y_a = anchors[...,1] + w_a*0.5

    # finding x, y, h, w 
    h = ground_truth[...,2] - ground_truth[...,0]
    w = ground_truth[...,3] - ground_truth[...,1]
    x = ground_truth[...,0] + h*0.5
    y = ground_truth[...,1] + w*0.5

    if ground_truth.size(0) != anchors.size(0):
        t_y = (y.unsqueeze(-1)-y_a)/h_a
        t_x = (x.unsqueeze(-1)-x_a)/w_a
        t_w = torch.log(w.unsqueeze(-1)/w_a)
        t_h = torch.log(h.unsqueeze(-1)/h_a)
    else:
        t_y = (y-y_a)/h_a
        t_x = (x-x_a)/w_a
        t_w = torch.log(w/w_a)
        t_h = torch.log(h/h_a)

    t = torch.stack((t_y,t_x,t_h, t_w), dim=-1)
    return t


def locToBox(anchors: torch.Tensor, locs: torch.Tensor):
    '''converts anchors of (y1,x1,y2,x2) format to (x,y,h,w) format

    Parameters
    ----------
    anchors : torch.Tensor ([N,4])
    locs : torch.Tensor ([B,N,4])

    Returns
    ----------
    torch.Tensor (B,N,4)
    '''
    eps = torch.tensor(1e-6)

    h_a = anchors[..., 2] - anchors[..., 0]
    w_a = anchors[..., 3] - anchors[..., 1]
    y_a = anchors[..., 0] + 0.5 * h_a
    x_a = anchors[..., 1] + 0.5 * w_a

    t_y = locs[...,0::4]     
    t_x = locs[...,1::4]
    t_h = torch.max(torch.exp(locs[...,2::4]), eps)
    t_w = torch.max(torch.exp(locs[...,3::4]), eps)

    y = t_y * h_a.unsqueeze(-1) + y_a.unsqueeze(-1)
    x = t_x * w_a.unsqueeze(-1) + x_a.unsqueeze(-1)
    h = t_h * h_a.unsqueeze(-1)
    w = t_w * w_a.unsqueeze(-1)

    rpn_boxes = torch.empty_like(locs)
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

def iou(groundtruth: torch.Tensor, anchor: torch.Tensor):
    if anchor.dim()==3:
        ious = list()
        split = torch.unique(groundtruth[:,0], return_counts=True)
        gt = torch.split(groundtruth, tuple(split[1]))
        ious = []
        for i, g in enumerate(gt):
            iou_g = calculate_iou(g, anchor[i,...])
            ious.append(iou_g)
        ious = torch.concat(ious)
    if anchor.dim()==2:
        ious = calculate_iou(groundtruth, anchor)
    return ious

def calculate_iou(groundtruth: torch.Tensor, anchor: torch.Tensor):
    '''Calculates the intersection-over-union between groundtruth anchors and the generated anchors

    Parameters
    ----------
    groundtruth : torch.Tensor [N,5] or [,5]
        _description_
    anchor : torch.Tensor [A,4]
        _description_

    Returns
    -------
    torch.Tensor [N,A]
        _description_
    '''
    if groundtruth.size(-1)==5:
        if groundtruth.dim()==1:
            groundtruth = groundtruth[1:]
        elif groundtruth.dim()==2:
            groundtruth = groundtruth[:,1:].unsqueeze(1)
    
    if anchor.size(-1) == 5:
        anchor = anchor[:,1:]
    

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
    Calculate multiple ious of groundtruth and anchors at the same time. The generated anchors are computed against each groundtruth bounding box.

    Parameters
    ----------
    groundtruth : [B, N, 4] torch.Tensor  
    anchors : [B, M, 4] torch.Tensor 
       
    where   B = batch_size
            N = number of ground truth bounding boxes (some observations are padded)
            M = number of anchors 
    Returns
    -------
    [B, M, N] torch.Tensor
        
    '''
    epsilon = 1e-6
    anchors_cut_area = (anchors[...,2]-anchors[...,0]) * (anchors[...,3]-anchors[...,1])
    gt_area = (groundtruth[...,2]-groundtruth[...,0]) * (groundtruth[...,3]-groundtruth[...,1])

    maxx = torch.max(anchors.unsqueeze(2).permute(1,0,2,3)[...,:2], groundtruth[...,:2]).permute(1,2,0,3)
    minn = torch.min(anchors.unsqueeze(2).permute(1,0,2,3)[...,2:], groundtruth[...,2:]).permute(1,2,0,3)

    intersect = torch.clip(minn[...,0]-maxx[...,0],0) * torch.clip(minn[...,1]-maxx[...,1],0)
    union = (anchors_cut_area.unsqueeze(1)+ gt_area.unsqueeze(-1))-intersect
    union = torch.clamp(union, min=epsilon)
    iou = (intersect/union).permute(0,2,1)
    return iou

def sampling(labels: torch.Tensor, n_sample: int=128, pos_ratio: float=0.5):
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

    n_pos = int(n_sample * pos_ratio)

    for i,label in enumerate(labels):
        pos_idx = torch.where(label==1)[0].float()
        if len(pos_idx) > n_pos:
            discard = torch.randperm(len(pos_idx))[:n_pos]
            discard = pos_idx[discard].long()
            labels[i,discard] = -1
        
        pos_idx = torch.where(labels[i,:]==1)[0].float()
        n_neg = n_sample-len(pos_idx)
        neg_idx = torch.where(label==0)[0].float()
        if len(neg_idx) > n_neg:
            discard = torch.randperm(len(neg_idx))[:n_neg]
            discard = neg_idx[discard].long()  
            labels[i,discard] = -1


    return labels
  

def assignLabels(ious, label, pos_iou_thres=0.5):
    '''
    Parameters
    ----------
    label : torch.Tensor
        [num_images, num_anchors] of -1 filled tensors
    ious : torch.Tensor
        [num_labels, num_anchors] 
    pos_ious_thres : float, optional
        label=1 if above the threshold and 0 if below 1-threshod, by default 0.5
    
    Returns
    -------
    [batch_size, num_obs] label:torch.Tensor
    '''
    
    label[ious>=pos_iou_thres] = 1
    label[ious<1-pos_iou_thres] = 0

    label_max_ious,_ = torch.max(ious, dim=1)
    all_label_max_idx = torch.where(ious==label_max_ious.unsqueeze(1))
    label[all_label_max_idx] = 1

    _,gt_idx = torch.max(ious, dim=0)   # gt_idx corresponds to which box the anchor has highest correspondence with

    return label, gt_idx 

def generateClsLab(fg_label, gt_label, gt_idx):
    gt_label = gt_label[:,1].clone()

    cls_label = torch.empty_like(fg_label)
    cls_label.fill_(0)

    selected = gt_label[gt_idx] # object label for each anchor
    selected = selected.unsqueeze(0).repeat(fg_label.size(0),1).type_as(cls_label)
    mask = fg_label.eq(1)

    cls_label[mask] = selected[mask] # to have the same dimension as output of score: [rpn_boxes, num_classes]
    return cls_label
    
def maxInputs(inputs, ious, splits):
    '''Reduces dimension of inputs, used when there are more than one groundtruth label in the image.
    
    Parameters
    ----------
    inputs : torch.tensor 
        _description_
    ious : torch.tensor
        _description_
    splits : tuple
        _description_

    Returns
    -------
    torch.tensor
        each batch consists of all labels possible given the iou
    '''
    inputs = torch.split(inputs, tuple(splits[1]))
    ious = torch.split(ious, tuple(splits[1]))
    outputs = list()    
    for input, iou in zip(inputs, ious):
        id = iou.max(0)[1]
        if input.dim()==3:
            range = torch.arange(iou.size(1))
            input = input[id, range]
        if input.dim()==2:
            if input.size(-1) != iou.size(-1):
                input = input[id]
            else:
                range = torch.arange(iou.size(1))
                input = input[id, range] 
            
        outputs.append(input)
    outputs = torch.stack(outputs)
    return outputs