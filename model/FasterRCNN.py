import torch
import torch.nn as nn
from model.FeatureExtractor import FeatureExtractor, Classifier
from . import losses
from torchvision.ops import RoIPool

class FasterRCNN(nn.Module):
    def __init__(self, base_architecture, rpn,config =None):
        super(FasterRCNN, self).__init__()       

        self.CONFIG = config
        self.training = config.training

        self.feat_stride = config.feat_stride
        self.num_classes = config.num_classes

        self.extractor = FeatureExtractor(base_architecture.extractor)
        self.rpn = rpn

        self.pooling = RoIPool(output_size=(config.H_roi_pool, config.W_roi_pool), spatial_scale=(1/self.feat_stride)) 
        self.clsifier = Classifier(base_architecture.classifier, config=config)

    def forward(self,im,gt_bboxes=None, obj_label=None):            # [N,3,H,W]
        feature_map = self.extractor(im)       # [N,512,H//16, W//16]
        proposals, rpn_loss = self.rpn(feature_map, gt_bboxes, obj_label)

        rpn_boxes, rpn_scores, rpn_indices  = proposals['boxes'], proposals['scores'],proposals['indices']

        # pooling
        sampled_boxes = rpn_boxes.view(-1,4)
        idx = rpn_indices.view(-1,1)

        pool_input = torch.cat([idx, sampled_boxes],dim=-1)
        pool_input[:,1:] = pool_input[:,1:].mul_(1/self.feat_stride)

        pool_output = self.pooling(feature_map, pool_input)
        pool_output = pool_output.view(pool_output.size(0),-1)

        # classifier
        loc, score = self.clsifier(pool_output)
        loc = loc.view(-1, self.num_classes, 4)

        rcnn_loss = {'reg_loss': 0, 'cls_loss':0}
        if self.training:
        # losses
            cls_loss, reg_loss = losses.fastrcnn_loss_fn(loc, score, proposals['locs'].clone(), proposals['cls_labels'].clone())
            rcnn_loss['reg_loss'] = reg_loss
            rcnn_loss['cls_loss'] = cls_loss 

        output = [rpn_boxes, rpn_scores, rpn_loss, loc, score, rcnn_loss, proposals]
        return output
