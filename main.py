import torch
from model.FasterRCNN import FasterRCNN
from data.Dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from data.utils import collate_fn
from config import Config
from model import losses, utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG = Config(TRAIN=True)
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

frcnn = FasterRCNN(config=CONFIG)
# frcnn.cuda(device)
print('done with f-rcnn')

with torch.no_grad():
    for images, labels, bboxs in train_loader:
        # images, labels,bboxs = images.to(device), labels.to(device), bboxs.to(device)
        outputs = frcnn(images,bboxs, labels)
        
        box_reg, obj_score, \
        rpn_rois, rpn_scores, rpn_labels, rpn_cls_label, rpn_indices, \
        target_anchor, target_labels, target_cls_label, \
        pool_input, pool_output, loc, score = outputs

        target_loc = utils.boxToLocation(frcnn.rpn.rpn_gt_roi, rpn_rois)

        rpn_cls_loss, rpn_reg_loss = losses.rpn_loss_fn(box_reg, obj_score, target_anchor, target_labels)
        cls_loss, reg_loss = losses.fastrcnn_loss_fn(loc, score, target_loc, rpn_cls_label)        

        print(f'rpn cls loss {rpn_cls_loss}, rpn reg_loss {rpn_reg_loss}')
        print(f'rnn cls loss {cls_loss}, rnn reg_loss {reg_loss}')
        





