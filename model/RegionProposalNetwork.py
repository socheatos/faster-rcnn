from torch.nn import functional as F
import torch
from torch import gt, nn
from . import utils, losses
from config import Config
from torch.nn.utils.rnn import pad_sequence
from math import sqrt


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# [x]: update self.CONFIG.batch_size manually for each input :( 
# [x]: some reason sampling returns 0 and -1s only not 1s DONE 
# [ ]: get rid of for loop for batch calculations
# [x]: find all anchors with same iou as the highest iou
# [ ]: add padding to support varying size of generated proposal 
# [x]: create separate modules for anchor and target generation
class AnchorGenerator():
    def __init__(self,config):
        self.ratios = config.ratios
        self.anchor_scales = config.anchor_scales
        self.height = config.input_h//config.feat_stride
        self.width = config.input_w//config.feat_stride
        self.feat_stride = config.feat_stride
        
    def generate(self):
        '''
        for each center, generate len(ratios)*len(anchor_scales) there
        there are centers_row*centers_col*len(ratios)*len(anchor_scales) in total
        for each center, generate the anchor boxes 
        '''
        centers = self.generateCenters()

        anchors = torch.zeros((len(centers)*len(self.anchor_scales)*len(self.ratios),4))
        idx = 0 
        for cntr_y, cntr_x in centers:
            for i in range(len(self.ratios)):
                for j in range(len(self.anchor_scales)):
                    h = self.feat_stride * self.anchor_scales[j] * sqrt(self.ratios[i])
                    w = self.feat_stride * self.anchor_scales[j] * sqrt(1/self.ratios[i])

                    anchors[idx, 0] = cntr_y - h /2
                    anchors[idx, 1] = cntr_x - w /2
                    anchors[idx, 2] = cntr_y + h /2
                    anchors[idx, 3] = cntr_x + w /2
                    
                    idx +=1

        return anchors

    def generateCenters(self):
        cntr_x = torch.arange(self.feat_stride, (self.width+1)*self.feat_stride, self.feat_stride) - self.width/2
        cntr_y = torch.arange(self.feat_stride, (self.height+1)*self.feat_stride, self.feat_stride) - self.height/2
        centers = torch.stack(torch.meshgrid((cntr_x,cntr_y), indexing='ij')).mT.reshape(-1,2)
        return centers

class ProposalGenerator(nn.Module):
    def __init__(self, config:Config) -> None:
        super(ProposalGenerator, self).__init__()
        self.anchor_base = config.anchor_base
        self.height = config.input_h
        self.width = config.input_w
        self.pre_nms = config.pre_nms
        self.post_nms = config.pos_nms
        self.nms_threshold = config.nms_threshold
        self.n_sample = config.proposal_n_sample
        self.pos_iou_threshold = config.proposal_pos_iou_thres
    
    def forward(self, anchor, box_reg, obj_score):
        batch_size = anchor.size(0)
        proposals = utils.locToBox(anchor, box_reg)
        proposals = self.clip(proposals)

        proposals, obj_score = self.filter(proposals, obj_score)

        rpn_roi = list()
        rpn_scores = list()
        for b in range(batch_size):
            roi = proposals[b]
            sco = obj_score[b]

            order = sco.ravel().argsort(descending=True)[:self.pre_nms]
            roi = roi[order,:]
            sco = sco[order]

            keep = utils.nms(roi, sco, self.nms_threshold)[:self.post_nms]
            roi = roi[keep]
            sco = sco[keep]

            rpn_roi.append(roi)
            rpn_scores.append(sco)
        
        # [ ] DONT pad, add the indices to indicate image instead 
        rpn_roi = pad_sequence(rpn_roi, batch_first=True)
        rpn_scores = pad_sequence(rpn_scores, batch_first=True)
        roi_indices = torch.ones_like(rpn_scores)
        for i in range(batch_size):
            roi_indices[i] = roi_indices[i]*i

        output = {'boxes': rpn_roi, 'scores': rpn_scores, 'indices': roi_indices}

        return output

    def assignLabels(self, boxes, scores, gt_boxes, obj_labels):
        # called only if training
        batch_size = boxes.size(0)
        labels = torch.empty_like(scores)
        labels.fill_(-1)

        ious = utils.iou_batch(gt_boxes, boxes)
        labels, gt_idx = utils.assignLabels(labels, ious,self.pos_iou_threshold)

        for b in range(batch_size):
            labels[b] = utils.sampling(labels[b],self.n_sample)

        gt_roi = utils.generateBBox(gt_boxes, gt_idx)
        gt_label = utils.generateClsLab(labels, obj_labels, gt_idx) # cls label

        output = {'boxes': gt_roi, 'labels':gt_label}
        return output

    def clip(self, boxes):
        boxes[...,0::2] = torch.clip(boxes[...,0::2], min=0, max=self.height)
        boxes[...,1::2] = torch.clip(boxes[...,1::2], min=0, max=self.width)
        return boxes

    def filter(self, boxes, scores):
        height = boxes[...,2] - boxes[...,0]
        width = boxes[...,3] - boxes[...,1]

        keep = torch.where((height>=self.anchor_base) & (width>=self.anchor_base))

        # [ ] DONT pad, add the indices to indicate image instead 
        boxes = pad_sequence(utils.split(boxes, keep), batch_first=True)
        scores = pad_sequence(utils.split(scores, keep), batch_first=True)

        return boxes, scores

class TargetGenerator(nn.Module):
    def __init__(self, config: Config):
        super(TargetGenerator, self).__init__()
        self.pos_ious_threshold = config.anchor_pos_iou_thres
        self.n_sample = config.anchor_n_sample
        self.height = config.input_h
        self.width = config.input_w
    
    def forward(self, anchors, gt_boxes, obj_labels):
        batch_size = anchors.size(0)
        index_inside = torch.where(
                                    (anchors[..., 0] >= 0) &
                                    (anchors[..., 1] >= 0) &
                                    (anchors[..., 2] <= self.height) &
                                    (anchors[..., 3] <= self.width))
        
        valid_anchors = anchors[index_inside]
        valid_anchors = valid_anchors.view(batch_size,-1,4)
        
        label = torch.empty((batch_size, anchors.size(1)))
        label.fill_(-1)
        # labels wrt to gt
        valid_ious = utils.iou_batch(gt_boxes, valid_anchors)
        
        ious = torch.zeros(batch_size, anchors.size(1), valid_ious.size(-1)).type_as(valid_ious)
        ious[index_inside] = valid_ious.reshape(-1, valid_ious.size(-1))
        self.target_ious = ious
        
        label, gt_idx = utils.assignLabels(label,ious, self.pos_ious_threshold)

        for b in range(batch_size):
            label[b] = utils.sampling(label[b],self.n_sample)

        gt_boxes = utils.generateBBox(gt_boxes, gt_idx)
        cls_label = utils.generateClsLab(label, obj_labels, gt_idx)

        anchors = utils.boxToLocation(gt_boxes, anchors)
        
        output = {'boxes': gt_boxes, 'anchors': anchors, 'labels':label, 'cls_labels': cls_label}

        return output


 
                   
class RegionProposalNetwork(nn.Module):
    '''
    A Region Proposal Network (RPN) takes feature maps as input and outputs a set of rectangular object proposals (RPN locs) 
    and objectness score (foreground or background).


    Parameters:
    ----
    :parameter:`in_channels`:   int
        channel size of feature map input
    :parameter:`mid_channels`:  int 
        channel size of immediate tensor output

    '''
    def __init__(self, anchor_generator, proposal_generator, target_generator,in_channels=512, mid_channels=512, config: Config=None):
        super(RegionProposalNetwork, self).__init__()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.CONFIG = config
        self.training = config.training
        self.anchor_scales = config.anchor_scales
        self.ratios = config.ratios   
        self.n_anchors = len(self.anchor_scales) * len(self.ratios)

        self.anchors = anchor_generator.generate().to(device)
        self.proposal_generator = proposal_generator
        self.target_generator = target_generator
        # intermediate layer extracting features from the feature map for proposal 
        # generation
        self.conv1 =  nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                                     kernel_size=3, stride=1, padding=1)

        # predicts box location
        self.reg =         nn.Conv2d(in_channels=mid_channels, out_channels=self.n_anchors*4,
                                      kernel_size=1, stride=1, padding=0)

        # every position on the feature map has 9 anchors, each has two possible labels (foreground, background)
        # we set the depth as 9*2 - every anchor will have a vector of 2 values (logit)
        # labels can be predicted if logit is fed into a softmax/logistic regression activation function
        self.cls =        nn.Conv2d(in_channels=mid_channels, out_channels=self.n_anchors*2,
                                      kernel_size=1, stride=1, padding=0)

        # initialize weights
        self.init_weight_n_biases()

        

    def forward(self,feat_map, gt_boxes=None, obj_label=None):
        '''
        Region proposals are generated by sliding a small network over the feature map whose input
        is an `3x3` spatial window of the input feature map. Then feature is fed into two sibling
        convolutional layer `(1 x 1)`- box-regression layer and box-classification layer. 

        Parameters:
        ---
        :parameter:`x`:          ~torch.Tensor :math:`[N,in_channels,H,W]`
            feature map extracted  from the input image
        '''
        self.batch_size, _,height,width = feat_map.shape
        feat_map = F.relu(self.conv1(feat_map), inplace=True)   # [N,out_channels,H,W]
        anchors =  self.anchors.view(1, self.anchors.size(0), 4).expand(self.batch_size, self.anchors.size(0), 4)

        box_reg = self.reg(feat_map)       # anchor location predictions    [N,A*4,H,W], deltas
        box_reg = box_reg.permute(0,2,3,1).contiguous().view(self.batch_size, -1, 4)   # reshape to same shape as anchor [N,W*H*A,4]       
    
        # print('box reg shape',box_reg.shape)
        obj_score = self.cls(feat_map)     # objectness score      [N,A*2,H,W]
        obj_fg_score = F.softmax(obj_score.view(self.batch_size, height,width,self.n_anchors,2),dim=-1)
        obj_fg_score = obj_fg_score[...,1].contiguous().view(self.batch_size,-1)

        obj_score = obj_score.permute(0,2,3,1).contiguous().view(self.batch_size, -1,2) #[N,H,W,A*2] -> [N, W*H*A, 2]

        targets = None
        gt_proposals = None
        loss = {'reg_loss': 0, 'cls_loss': 0}
        
        proposals = self.proposal_generator(anchors,box_reg, obj_fg_score)

        if self.training:
            targets = self.target_generator(anchors, gt_boxes, obj_label)

            gt_proposals = self.proposal_generator.assignLabels(proposals['boxes'], proposals['scores'], gt_boxes, obj_label)

            proposals['locs'] = utils.boxToLocation(gt_proposals['boxes'], proposals['boxes'])
            proposals['cls_labels'] = gt_proposals['labels']

            target_anchors = targets['anchors']
            target_labels = targets['labels']

            cls_loss, reg_loss = losses.rpn_loss_fn(box_reg, obj_score, target_anchors, target_labels)

            loss['reg_loss'] = reg_loss
            loss['cls_loss'] = cls_loss
        
        return proposals, loss

    def init_weight_n_biases(self):
        self.conv1.weight.data.normal_(0,0.1)
        self.conv1.bias.data.zero_()

        self.reg.weight.data.normal_(0,0.1)
        self.reg.bias.data.zero_()

        self.cls.weight.data.normal_(0,0.1)
        self.cls.bias.data.zero_()
    
 
