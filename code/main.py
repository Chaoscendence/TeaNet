import argparse
import torch
import os
import time
from configs import cfg
from data_handle.dataset import dataset_divide
from data_handle.dataset import dataset_mask
from train_model import set_config
from train_model import train_TEA
from test_model import test_TEA
from utils.log import get_logger

DATASET_DIR = "results"
parser = argparse.ArgumentParser(description='TeaNet')
parser.add_argument(
    "--config",
    default='configs/config.yaml',
    help="path to config file",
    type=str
)

parser.add_argument('--train', dest='train', action='store_true', help='flag to start training')
parser.set_defaults(train=False)
parser.add_argument('--test', dest='test', action='store_true', help='run test')
parser.set_defaults(test=False)
parser.add_argument('--idx_gpu', dest='idx_gpu', help='idx of gpu using for training')
parser.set_defaults(idx_gpu=0)
parser.add_argument('--models_dir', dest='models_dir', help='path of testing model')
parser.set_defaults(models_dir=None)
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.freeze()
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
save_path = os.path.join(DATASET_DIR, time_now)

settings = {
    # TRAIN
    'cuda': cfg.TRAIN.cuda,
    'kernel': cfg.TRAIN.kernel,
    'kernel_unet': cfg.TRAIN.kernel_unet,
    'width': cfg.TRAIN.width,
    'epochs': cfg.TRAIN.epochs,
    'verbose': cfg.TRAIN.verbose,
    'batch_size': cfg.TRAIN.batch_size,
    'lr_base_G': cfg.TRAIN.lr_base_G,
    'lr_base_D': cfg.TRAIN.lr_base_D,
    'loss_ratio': cfg.TRAIN.loss_ratio,
    'ConvLayers': cfg.TRAIN.ConvLayers,
    'linear_dim': cfg.TRAIN.linear_dim,
    'class_num': cfg.TRAIN.class_num,
    # DATASET
    'mask_scheme': cfg.DATASET.mask_scheme,
    'mask_num': cfg.DATASET.mask_num,
    'data_path': cfg.DATASET.data_path,
    'scheme': cfg.DATASET.scheme
}

def main():
    data = torch.load(settings['data_path'])
    if args.train:
        print('----------------------------------------------------------')
        print('Notice: split the dataset')
        print('----------------------------------------------------------')
        data = dataset_divide(settings['scheme'], data)
        print('----------------------------------------------------------')
        print('Notice: augment the dataset')
        print('----------------------------------------------------------')
        data = dataset_mask(data, settings['mask_scheme'], settings['mask_num'], None)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logger = get_logger(log_file_name=os.path.join(save_path, 'log'))
        logger.info(settings)
        train_TEA(data, save_path, logger)
    if args.test:
        if args.models_dir == None:
            results = os.listdir(DATASET_DIR)
            results = sorted(results)
            args.models_dir = os.path.join(DATASET_DIR, results[-1])
        test_TEA(data, settings, args.models_dir, int(args.idx_gpu))
        file = open(os.path.join(args.models_dir, 'settings.txt'), 'w')
        file.write(str(settings))
    print('Nothing will happen. Have fun!')

if __name__ == "__main__":
    set_config(settings, int(args.idx_gpu))
    main()
