import torchvision
import torch.nn as nn
import torch
# vgg = torchvision.models.vgg16(pretrained=True)
# features = list(vgg.features[:30])
# classifier = list(vgg.classifier)
# extractor = nn.Sequential(*features)

class BaseArchitect():
    def __init__(self, config=None):
        try:
            vgg = torchvision.models.vgg16()
            vgg._load_from_state_dict(torch.load('data/vgg16-397923af.pth'))
        except:
            vgg = torchvision.models.vgg16(pretrained=True)
        features = list(vgg.features[:30])
        for f in features[:24]:
            f.trainable = False # Freeze the convolution blocks for finetuning 
        classifier = list(vgg.classifier[:-1])

        self.CONFIG = config

        self.extractor = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)


class FeatureExtractor(nn.Module):
    def __init__(self, feature_ext):
        super().__init__()
        self.extractor = feature_ext

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
    
class Classifier(nn.Module):
    def __init__(self, classifier, config=None):
        super().__init__()
        self.CONFIG = config

        self.classifier = classifier
        self.cls_loc = nn.Linear(in_features=4096, out_features=config.num_classes*4)
        self.cls_score = nn.Linear(in_features=4096, out_features=config.num_classes)

        self.init_weights()

    def forward(self,x):
        x = self.classifier(x)
        loc = self.cls_loc(x)
        score = self.cls_score(x)
        
        return loc, score

    def init_weights(self):
        self.cls_loc.weight.data.normal_(0,0.01)
        self.cls_loc.bias.data.zero_()
        self.cls_score.weight.data.normal_(0,0.01)
        self.cls_score.bias.data.zero_()

