from torch.nn import functional as F
import torch
from torch import nn
from . import losses, utils
from math import sqrt
from config import Config
from torch.nn.utils.rnn import pad_sequence


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

class ProposalGenerator():
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
    
    def __call__(self, anchor, box_reg, obj_score):
        proposals = utils.locToBox(anchor, box_reg) # ([B,N,4]; [B,N,4]) -> ([N,4] ; [B,N,4])
        proposals = self.clip(proposals)

        ones = torch.zeros((proposals.size(0), proposals.size(1),1))  
        proposals = torch.concat((ones, proposals),dim=-1)

        proposals, obj_score = self.filter(proposals, obj_score)

        rpn_rois = list()
        rpn_scores = list()
        for box,score in zip(proposals, obj_score):
            order = score.ravel().argsort(descending=True)[:self.pre_nms]
            box = box[order,:]
            score = score[order]

            keep = utils.nms(box, score, self.nms_threshold)[:self.post_nms]
            box = box[keep,:]
            score = score[keep]

            rpn_rois.append(box)
            rpn_scores.append(score)
        
        rpn_rois = pad_sequence(rpn_rois, batch_first=True)
        rpn_scores = pad_sequence(rpn_scores, batch_first=True)
        
        if rpn_rois.dim()==3:
            for i in range(rpn_rois.size(0)):
                rpn_rois[i,:,0] = i

        output = {'boxes': rpn_rois, 'scores': rpn_scores}

        return output

    def assignLabels(self, boxes, scores, gt_boxes, obj_labels, num_classes=None):
        # called only if training
        
        label_idx = torch.unique(gt_boxes[:,0], return_counts=True)
        ious = utils.iou(gt_boxes, boxes)
        labels = torch.empty_like(ious)
        labels.fill_(-1)

        labels, gt_idx = utils.assignLabels(ious, labels,self.pos_iou_threshold)
        labels = utils.maxInputs(labels, ious, label_idx)
        labels = utils.sampling(labels, self.n_sample).type_as(scores)

        # gt_roi = gt_boxes[gt_idx].type_as(gt_boxes)
        gt_roi = utils.maxInputs(gt_boxes, ious, label_idx).type_as(gt_boxes)
        gt_label = utils.generateClsLab(labels, obj_labels, gt_idx).type_as(obj_labels) # cls label need to go from [G,N] -> [NumClasses, N]

        output = {'boxes': gt_roi, 'labels':gt_label, 'idx':gt_idx}
        return output

    def clip(self, boxes):
        boxes[...,0::2] = torch.clip(boxes[...,0::2], min=0, max=self.height)
        boxes[...,1::2] = torch.clip(boxes[...,1::2], min=0, max=self.width)
        return boxes

    def filter(self, boxes, scores):
        height = boxes[...,2] - boxes[...,0]
        width = boxes[...,3] - boxes[...,1]

        keep = torch.where((height>=self.anchor_base) & (width>=self.anchor_base))

        boxes = utils.split(boxes, keep)
        scores = utils.split(scores,keep)

        return boxes, scores


class TargetGenerator():
    def __init__(self, config: Config):
        super(TargetGenerator, self).__init__()
        self.pos_ious_threshold = config.anchor_pos_iou_thres
        self.n_sample = config.anchor_n_sample
        self.height = config.input_h
        self.width = config.input_w
    
    def __call__(self, anchors, gt_boxes, gt_labels):
        label_idx = torch.unique(gt_boxes[:,0], return_counts= True)
        
        # batch_size = label_idx[0].size(0)
        index_inside = torch.where(
                                    (anchors[..., 0] >= 0) &
                                    (anchors[..., 1] >= 0) &
                                    (anchors[..., 2] <= self.height) &
                                    (anchors[..., 3] <= self.width))
        
        valid_anchors = anchors[index_inside]
        
        valid_ious = utils.iou(gt_boxes, valid_anchors)  
        ious = torch.zeros((gt_boxes.size(0), anchors.size(0))).type_as(valid_ious)
        ious[:, index_inside[0]] = valid_ious
        
        label = torch.empty_like(ious)
        label.fill_(-1)

        label, gt_idx = utils.assignLabels(ious, label,self.pos_ious_threshold)
        label = utils.maxInputs(label, ious, label_idx)
        label = utils.sampling(label, self.n_sample).type_as(gt_labels)  # [G, N]
        
        gt_box = utils.maxInputs(gt_boxes, ious, label_idx).type_as(gt_boxes)
        cls_label = utils.generateClsLab(label, gt_labels, gt_idx).type_as(gt_labels)

        locs = utils.boxToLoc(gt_box, anchors).type_as(gt_boxes)
        
        output = {'boxes': gt_box, 'locs': locs, 'labels':label, 'cls_labels': cls_label}

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
    def __init__(self,in_channels=512, mid_channels=512, config: Config=None):
        super(RegionProposalNetwork, self).__init__()
        
        self.training = config.training
        self.anchor_scales = config.anchor_scales
        self.ratios = config.ratios 
        self.num_classes = config.num_classes  
        self.n_anchors = len(self.anchor_scales) * len(self.ratios)
        
        self.anchors = AnchorGenerator(config).generate()
        self.proposal_generator = ProposalGenerator(config)
        self.target_generator = TargetGenerator(config)

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

    def forward(self,feat_map, gt_boxes=None, gt_labels=None):
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
        conv1 = F.relu(self.conv1(feat_map), inplace=False)   # [B,out_channels,H,W]
        # anchors =  self.anchors.view(1, self.anchors.size(0), 4).expand(self.batch_size, self.anchors.size(0), 4).type_as(conv1)
        anchors = self.anchors
        
        box_reg = self.reg(conv1)       # anchor location predictions    [B,A*4,H,W], deltas
        box_reg = box_reg.permute(0,2,3,1).contiguous().view(self.batch_size, -1, 4)   # reshape to same shape as anchor [B,W*H*A,4]       
    
        # print('box reg shape',box_reg.shape)
        obj_score = self.cls(conv1)     # objectness score      [B,A*2,H,W]
        obj_score = obj_score.permute(0,2,3,1).contiguous()
        obj_fg_score = F.softmax(obj_score.view(self.batch_size, height,width,self.n_anchors,2),dim=-1)
        obj_fg_score = obj_fg_score[...,1].contiguous().view(self.batch_size,-1)

        obj_score = obj_score.view(self.batch_size, -1,2) #[B,H,W,A*2] -> [B, W*H*A, 2]
        loss = {'reg_loss': 0, 'cls_loss': 0}

        # {'boxes': rpn_rois, 'scores': rpn_scores}
        proposals = self.proposal_generator(anchors,box_reg.detach(), obj_fg_score.detach()) 
        self.proposals = proposals
        # print('rpn bxoes rpn', proposals['boxes'].shape)
        if self.training:
            targets = self.target_generator(anchors, gt_boxes, gt_labels)

            gt_proposals = self.proposal_generator.assignLabels(proposals['boxes'], proposals['scores'], gt_boxes, gt_labels)
            # print(gt_proposals['boxes'].shape, proposals['boxes'].shape)

            proposals['locs'] = utils.boxToLoc(gt_proposals['boxes'], proposals['boxes'])
            proposals['gt_boxes'] = gt_proposals['boxes']
            proposals['cls_labels'] = gt_proposals['labels']

            target_locs = targets['locs']
            target_labels = targets['labels'].type_as(gt_labels)
            
            cls_loss, reg_loss = losses.rpn_loss_fn(box_reg.detach(), obj_score.detach(), target_locs, target_labels)

            loss['reg_loss'] = reg_loss
            loss['cls_loss'] = cls_loss

            self.proposals = proposals
            self.targets = targets
            self.gt_proposals = gt_proposals
        return proposals, targets, loss



    def init_weight_n_biases(self):
        self.conv1.weight.data.normal_(0,0.1)
        self.conv1.bias.data.zero_()

        self.reg.weight.data.normal_(0,0.1)
        self.reg.bias.data.zero_()

        self.cls.weight.data.normal_(0,0.1)
        self.cls.bias.data.zero_()
    
 
