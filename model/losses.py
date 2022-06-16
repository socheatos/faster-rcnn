import torch.nn.functional as F
import torch 
def loss_reg(predicted, groundtruth, label, lam=10):
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

    smooth_l1 = torch.where(diff_abs<1, 0.5*diff**2, diff-0.5)
    sum_smooth_l1 = torch.sum(smooth_l1, dim=2)
    print(len(torch.where(sum_smooth_l1<0)[0]))
    
    zeros = torch.zeros_like(sum_smooth_l1)
    l_reg = torch.sum(torch.where(label==1, sum_smooth_l1,zeros))
    loss = l_reg.mean() * lam

    return loss

def loss_cls(predicted, groundtruth):
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
    loss = F.cross_entropy(predicted, groundtruth, ignore_index=-1)
    return loss