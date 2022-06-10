import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(self, feat_ext, rpn):
        super().__init__()
        self.extractor = feat_ext
        self.rpn = rpn

    def forward(self,x,gt_bboxes=None):            # [N,3,H,W]
        x = self.extractor(x)       # [N,512,H//16, W//16]
        x,y= self.rpn(x, gt_bboxes)            
        return x,y