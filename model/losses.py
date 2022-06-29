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