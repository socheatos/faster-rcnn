import torch
import torch.nn as nn
import torch.nn.functional as F
from model import losses, utils
from model.RegionProposalNetwork import RegionProposalNetwork 
from model.FeatureExtractor import BaseArchitect, FeatureExtractor, Classifier
from torchvision.ops import RoIPool

# [ ] assign class labels 

class FasterRCNN(nn.Module):
    def __init__(self, config =None):
        super().__init__()
        vgg = BaseArchitect(config=config)
        self.CONFIG = config
        self.extractor = FeatureExtractor(vgg.extractor)

        self.rpn = RegionProposalNetwork(config=config)

        self.pooling = RoIPool(output_size=(config.H_roi_pool, config.W_roi_pool), spatial_scale=(1/config.feat_stride)) 
        self.clsifier = Classifier(vgg.classifier, config=config)

    def forward(self,im,gt_bboxes=None, obj_label=None):            # [N,3,H,W]
        feature_map = self.extractor(im)       # [N,512,H//16, W//16]

        box_reg, obj_score, \
        rpn_rois, rpn_scores, rpn_labels, rpn_cls_label, rpn_indices, \
        target_anchor, target_labels, target_cls_label = self.rpn(feature_map, gt_bboxes, obj_label)
        
        # feature map and proposals go into roi pooling layer
        # sampled_mask = rpn_labels.gt(-1)

        # sampled_rois = rpn_rois[sampled_mask].view(-1,4)
        # sampled_rois = utils.toXYXY(sampled_rois)
        # sampled_cls_label = rpn_cls_label[sampled_mask].long()
        # idx = rpn_indices[sampled_mask].view(-1,1)

        idx = rpn_indices.view(-1,1)
        sampled_rois = rpn_rois.view(-1,4)

        pool_input = torch.cat([idx, sampled_rois], dim=-1)
        pool_input[:,1:] =pool_input[:,1:].mul_(1/self.CONFIG.feat_stride)

        pool_output = self.pooling(feature_map, pool_input)
        pool_output = pool_output.view(pool_output.size(0),-1)

        # pass to the classifier
        loc, score = self.clsifier(pool_output)
        loc = loc.view(-1, self.CONFIG.num_classes,4)


        rpn_cls_idx = rpn_cls_label.flatten().long().unsqueeze(1).repeat(1,4).unsqueeze(1)
        # print(rpn_cls_label, rpn_cls_idx)
        loc_cls = torch.gather(loc,1,rpn_cls_idx).squeeze()
        rpn_roi_reg = utils.boxToLocation(self.rpn.rpn_gt_roi, rpn_rois)


        # loss
        cls_loss = losses.loss_cls(score, rpn_cls_label.flatten().long())
        reg_loss = losses.loss_reg(loc_cls, rpn_roi_reg.view(-1,4), rpn_cls_label.flatten(),1,False)

        print(f'rnn cls loss {cls_loss}, rnn reg_loss {reg_loss}')

        # fast rcnn loss
        output = [box_reg, obj_score,rpn_rois, rpn_scores, rpn_labels, rpn_cls_label,rpn_indices, target_anchor, target_labels,target_cls_label,\
                    pool_input, pool_output, loc, score]
        return output
