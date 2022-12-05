from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
_C.TRAIN.cuda = True
_C.TRAIN.kernel = 21
_C.TRAIN.kernel_unet = 9
_C.TRAIN.lr_base_D = 1e-4
_C.TRAIN.lr_base_G = 1e-3
_C.TRAIN.loss_ratio = 0.7
_C.TRAIN.width = 3
_C.TRAIN.epochs = 300
_C.TRAIN.verbose = True
_C.TRAIN.batch_size = 256
_C.TRAIN.ConvLayers = [16, 16, 'M']
_C.TRAIN.linear_dim = 8192
_C.TRAIN.class_num = 430

_C.DATASET = CN()
_C.DATASET.mask_scheme = [0.3, 0.7]
_C.DATASET.mask_num = 10
_C.DATASET.data_path = '../DATA/rruff_ir.pth'
_C.DATASET.scheme = 0.80

cfg = _C
