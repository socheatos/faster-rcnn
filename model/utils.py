import torch 
import numpy as np

def nms(dets, scores,thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

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

    return anchors.astype(np.int32)

def clip(box, thres):
    pass

def split(batch, keep):
    counts = torch.unique(keep[0],return_counts=True)[1]
    splits = tuple(counts.numpy())

    try:
        batch = batch[keep[0],keep[1],...]
    except:
        batch = batch[keep[0],keep[1]]
        
    batch = torch.split(batch, split_size_or_sections=splits)
    return batch



