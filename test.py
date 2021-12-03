import os
import time
import json
import logging
import argparse
import numpy as np

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import albumentations as A

cudnn.benchmark = True

import lib.core.utils as utils
from lib.core.config import config
from lib.core.config import update_config
from lib.core.lfw_eval import eval as lfw_eval
from lib.models.resnets import LResNet50E_IR
from lib.datasets.dataset import LFW_Image

# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch SynFace')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--frequent', help='frequency of logging', default=config.TRAIN.PRINT_FREQ, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--lr', help='init learning rate', type=float)
    parser.add_argument('--optim', help='optimizer type', type=str)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=0, type=int)
    parser.add_argument('--model', help=' model name', type=str)
    parser.add_argument('--loss_type', help='loss type', type=str)
    parser.add_argument('--focal', help='focal loss', default=0, type=int)
    parser.add_argument('--dataset', help=' training dataset name', default='WebFace', type=str)
    parser.add_argument('--syn', help='new syn root', default='', type=str)
    parser.add_argument('--dm', help='whether mixup with real data for training (i.e., domain mixup)', type=int)
    parser.add_argument('--num_id', help='num of id', default=10000, type=int)
    parser.add_argument('--samples_perid', help='samples per id', default=50, type=int)
    parser.add_argument('--batch_size', help='batch size', default=512, type=int)
    parser.add_argument('--real_num_id', help='num of real id', default=1000, type=int)
    parser.add_argument('--real_samples_perid', help='real samples per id', default=10, type=int)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.TRAIN.GPUS = args.gpus
    if args.workers:
        config.TRAIN.WORKERS = args.workers
    if args.model:
        print('update model type')
        config.TRAIN.MODEL = args.model
    if args.lr:
        print('update learning rate')
        config.TRAIN.LR = args.lr
    if args.pretrained =='No':
        print('update pretrained')
        config.NETWORK.PRETRAINED = ''
    if args.optim:
        print('update optimizer type')
        config.TRAIN.OPTIMIZER = args.optim
    if args.loss_type:
        config.LOSS.TYPE = {
            'Arc': 'ArcMargin',
            'Cos': 'CosMargin'
        }[args.loss_type]
    if args.syn:
        print('update syn root: {}'.format(args.syn))
        config.DATASET.SYN_ROOT = args.syn
    if args.dm:
        print('update dm')
        config.TRAIN.DM = args.dm
    config.TRAIN.NUM_ID =args.num_id 
    config.TRAIN.SAMPLES_PERID =args.samples_perid
    config.REAL.NUM_ID =args.real_num_id 
    config.REAL.SAMPLES_PERID =args.real_samples_perid

    config.TRAIN.BATCH_SIZE = args.batch_size
    config.TEST.BATCH_SIZE = args.batch_size
def main():
    # --------------------------------------model----------------------------------------
    args = parse_args()
    reset_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    logger, final_output_dir, tb_log_dir = utils.create_temp_logger()

    # ------------------------------------load image---------------------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    logger.info(test_transform)

    test_loader = torch.utils.data.DataLoader(
        LFW_Image(config.DATASET.LFW_PATH, config.DATASET.LFW_PAIRS, test_transform),
        batch_size=config.TEST.BATCH_SIZE * len(gpus), 
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS, 
        pin_memory=True)

    model = LResNet50E_IR(input_size=config.NETWORK.IMAGE_SIZE)

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    logger.info(model)
    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    model_root = 'models/'
    model_paths = ['model_10k_50_nomix_8898.pth.tar', 'model_best_2021-03-05-19-49_0.9765.pth.tar', 'model_best_2021-03-05-17-08_0.9578.pth.tar', 'model_10k_50_idmix_9197.pth.tar']

    for model_path in model_paths:
        logger.info(model_path)
        model_path = os.path.join(model_root, model_path)
        lfw_eval(model, model_path, config, test_loader, 'temp', 0)

if __name__ == '__main__':
    main()
