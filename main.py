import torch
from model.FasterRCNN import FasterRCNN
from data.Dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from data.utils import collate_fn
from config import Config
# from model import losses, utils
from model.FeatureExtractor import BaseArchitect
from model.RegionProposalNetwork import RegionProposalNetwork, AnchorGenerator, ProposalGenerator, TargetGenerator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG = Config(TRAIN=False)
# ------------ Dataset Prep -----------------
s= 600
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((CONFIG.input_h,CONFIG.input_w)),
                                transforms.ToTensor()])
dataset = ImageDataset('data/images/','data/labels_all_ano.csv',transform=transform)
train_loader = DataLoader(dataset,batch_size=CONFIG.batch_size,shuffle=True, collate_fn=collate_fn)

print(f'Batches in test_set {len(train_loader)}')
print(f'Examples in test_set {len(train_loader.sampler)}')


# # # # ------------ Model Load -----------------
base_architecture = BaseArchitect(CONFIG)
anchor_generator = AnchorGenerator(CONFIG)
proposal_generator = ProposalGenerator(CONFIG)
target_generator = TargetGenerator(CONFIG)
rpn = RegionProposalNetwork(anchor_generator, proposal_generator, target_generator,config=CONFIG)

frcnn = FasterRCNN(base_architecture,rpn,config=CONFIG)
# frcnn.cuda(device)
print('done with f-rcnn')

with torch.no_grad():
    for images, labels, bboxs in train_loader:
        # images, labels,bboxs = images.to(device), labels.to(device), bboxs.to(device)
        outputs = frcnn(images,bboxs, labels)
        