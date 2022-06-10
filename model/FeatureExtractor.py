from turtle import forward
import torchvision
import torch.nn as nn
import torch

# vgg = torchvision.models.vgg16(pretrained=True)
# features = list(vgg.features[:30])
# classifier = list(vgg.classifier)
# extractor = nn.Sequential(*features)

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = torchvision.models.vgg16()
            vgg._load_from_state_dict(torch.load('data/vgg16-397923af.pth'))
        except:
            vgg = torchvision.models.vgg16(pretrained=True)
        features = list(vgg.features[:30])
        self.extractor = nn.Sequential(*features)
        classifier = list(vgg.classifier)

    def forward(self,x):
        '''
        Forwards the batched images to the network
        Parameter
        ---
        :parameter:`x`  ~torch.Tensor size :math:`[N,C,H,W]` where
            :math:`N`:  batch size, :math:`C`:  channel, :math:`H`:  height, :math:`W`:  weight

        Returns
        ---
        :parameter:`x`  ~torch.Tensor size [N,512,H//16,W//16]
        '''
        x = self.extractor(x)
        return x
        

