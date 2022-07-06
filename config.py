class Config():
    def __init__(self,TRAIN=False):
        self.training = TRAIN
        self.input_h = 600
        self.input_w = 600
        self.batch_size = 2
        self.num_classes = 3+1
        
        # 'sub-sampling ratio'
        self.feat_stride = 16 # (original img dimensions)/(feature map dimension) for VGG it's 16
        self.H_roi_pool= 7
        self.W_roi_pool = 7

        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]
        # self.anchor_scales = [4, 8, 16]
        self.anchor_base = 16 # 16

        self.nms_threshold = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_pos_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_pos_nms = 300
        
        if TRAIN:
            self.pre_nms = self.n_train_pre_nms
            self.pos_nms = self.n_train_pos_nms
        else:
            self.pre_nms = self.n_test_pre_nms
            self.pos_nms = self.n_test_pos_nms

        self.pos_ratio = 0.5

        self.anchor_pos_iou_thres = 0.5
        self.anchor_n_sample = 256
        
        self.proposal_pos_iou_thres = 0.7
        self.proposal_n_sample = 128

