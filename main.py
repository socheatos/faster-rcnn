import torch
from model import RegionProposalNetwork, FeatureExtractor, FasterRCNN
from data.Dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from data.utils import collate_fn, collate_fn_2
from config import Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG = Config()
# ------------ Dataset Prep -----------------
s= 600
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((CONFIG.input_h,CONFIG.input_w)),
                                transforms.ToTensor()])
dataset = ImageDataset('data/images/','data/labels.csv',transform=transform)
train_loader = DataLoader(dataset,batch_size=2,shuffle=True)

print(f'Batches in test_set {len(train_loader)}')
print(f'Examples in test_set {len(train_loader.sampler)}')


# # # # ------------ Model Load -----------------

vgg = FeatureExtractor.VGG16()
rpn = RegionProposalNetwork.RegionProposalNetwork(config=CONFIG)
frcnn = FasterRCNN.FasterRCNN(vgg, rpn)
print('done with f-rcnn')

with torch.no_grad():
    for images, labels, bboxs in train_loader:
        images, labels,bboxs = images.to(device), labels.to(device), bboxs.to(device)
        bbox_output, score_output = frcnn(images,bboxs)
        # print(f'faster rcnn output: {bbox_output.shape, score_output.shape}')

        





