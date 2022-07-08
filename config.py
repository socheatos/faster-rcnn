class Config():
    def __init__(self, TRAIN=False,
                    device:str ='cpu',
                    optimizer:str = 'SGD',
                    epochs:int = 5, 
                    lr:float = .001,
                    momentum:float = 0.9,
                    weight_decay:float = 0.0005,
                    pretrained_model:str = None,
                    validate:bool =False,
                    
                    input_h:int = 600,
                    input_w:int = 600,
                    num_classes:int = 4,
                    batch_size:int = 4):
        # training config
        self.device = device
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr 
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.pretrained_model = pretrained_model
        self.validate = validate


        # model configurations
        self.training = TRAIN
        self.input_h = input_h
        self.input_w = input_w
        self.batch_size = batch_size
        self.num_classes = num_classes # num classes + background
        
        self.feat_stride = 16 # (original img dimensions)/(feature map dimension) for VGG it's 16
        self.H_roi_pool= 7
        self.W_roi_pool = 7

        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]
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



