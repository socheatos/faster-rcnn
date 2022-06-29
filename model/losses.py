import torch.nn.functional as F
import torch 
def loss_reg(predicted, groundtruth, label, lam=10, mean=True):
    '''
    Regression loss
    L_reg(t_i, t_i^*) = R(t_i - t_i^*)w where  is the robus losss function (smooth L1).
    :math:`smooth_L_1(x)` = 0.5x^2 (|x|<1) and |x|-0.5 otherwise
    
    Parameter
    ---
    :parameter:`predicted` parameterized locations of predicted RoI for the object 
    :parameter:`groundtruth` parameterized location of ground truth box
    '''
    diff = predicted-groundtruth
    diff_abs = torch.abs(diff)    

    smooth_l1 = torch.where(diff_abs<1, 0.5*diff**2, diff_abs-0.5).sum(dim=-1)
    zeros = torch.zeros_like(smooth_l1)

    loss = torch.where(label>0, smooth_l1,zeros).sum() * lam
    if mean:
        loss = torch.mean(torch.where(label==1, smooth_l1, zeros)) * lam

    return loss

def loss_cls(predicted, groundtruth, ignore_index=-1):
    '''
    Log-loss over two classes (object vs. not object)
    :math:`L_cls(p,u)`= - log(p_u)
    
    Parameter
    ---
    :parameter:`predicted` `~torch.Tensor` size: (num_batch, num_class)
        discrete probability distrubtion computed by softmax over the outptu of the rpn_cls layer
    :parameter:`groundtruth` `~torch.Tensor` size: (num_batch, num_class)
        ROI label (object: 1, background: 0)
    '''
    loss = F.cross_entropy(predicted, groundtruth, ignore_index= ignore_index)
    return loss

def rpn_loss_fn(predicted_locs, predicted_scores, gt_locs, gt_labels):
    cls_loss = loss_cls(predicted_scores.view(-1,2), gt_labels.flatten().long())
    reg_loss = loss_reg(predicted_locs, gt_locs, gt_labels)

    return cls_loss, reg_loss

def fastrcnn_loss_fn(loc, score, target_loc, target_cls):
    target_cls = target_cls.flatten().long()
    target_loc = target_loc.view(-1,4)

    rpn_cls_idx = target_cls.flatten().long().unsqueeze(1).repeat(1,4).unsqueeze(1)
    loc_cls = torch.gather(loc, 1, rpn_cls_idx).squeeze()
    
    cls_loss = loss_cls(score, target_cls)
    reg_loss = loss_reg(loc_cls, target_loc, target_cls,1,False)

    return cls_loss, reg_loss