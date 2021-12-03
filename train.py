import argparse
import os
import time
import shutil
import logging
import json
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
from lib.core.functions import train
from lib.core.lfw_eval import eval as lfw_eval
from lib.datasets.dataset import Img_Folder_Num
from lib.datasets.dataset import Img_Folder_Mix
from lib.datasets.dataset import LFW_Image
from lib.models.resnets import LResNet50E_IR
from lib.models.metrics import ArcMarginProduct
from lib.models.metrics import CosMarginProduct
from lib.models.loss import FocalLoss

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

    dataset_root = {
        'WebFace': config.DATASET.WEBFACE_ROOT,
        'Syn': config.DATASET.SYN_ROOT
    }[args.dataset]

    if args.debug:
        logger, final_output_dir, tb_log_dir = utils.create_temp_logger()
    else:
        logger, final_output_dir, tb_log_dir = utils.create_logger(
            config, args.cfg, dataset_root, 'train')

    # ------------------------------------load image---------------------------------------
    train_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomApply([transforms.Resize(112),
                                transforms.RandomCrop(config.NETWORK.IMAGE_SIZE)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  
        transforms.RandomErasing()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # albu_transform = A.Compose([
        # A.OneOf([
            # A.GaussianBlur(),
            # A.MotionBlur(),
            # A.MedianBlur(),
            # A.Blur(),
        # ], p=0.8),
        # A.OneOf([
            # A.OpticalDistortion(p=0.3),
            # A.GridDistortion(p=0.1),
        # ], p=0.5),
    # ])
    albu_transform = None
    logger.info(train_transform)
    logger.info(albu_transform)

    syn_dataset = Img_Folder_Mix(config.DATASET.SYN_ROOT, albu_transform, train_transform, num_id=config.TRAIN.NUM_ID, samples_perid=config.TRAIN.SAMPLES_PERID)
    real_dataset = Img_Folder_Num(config.DATASET.WEBFACE_ROOT, len(syn_dataset.classes), len(syn_dataset), albu_transform, train_transform, num_id=config.REAL.NUM_ID, samples_perid=config.REAL.SAMPLES_PERID)
    # pdb.set_trace()

    real_loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus), 
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.TRAIN.WORKERS, 
        pin_memory=True, 
        drop_last=True)

    syn_loader = torch.utils.data.DataLoader(
        syn_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus), 
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.TRAIN.WORKERS, 
        pin_memory=True, 
        drop_last=True)

    assert len(real_loader) == len(syn_loader)

    test_loader = torch.utils.data.DataLoader(
        LFW_Image(config.DATASET.LFW_PATH, config.DATASET.LFW_PAIRS, test_transform),
        batch_size=config.TEST.BATCH_SIZE * len(gpus), 
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS, 
        pin_memory=True)

    num_class = len(syn_dataset.classes) + len(real_dataset.classes) 

    model = LResNet50E_IR(input_size=config.NETWORK.IMAGE_SIZE)
    # choose the type of loss 512 is dimension of feature
    classifier = {
        'ArcMargin': ArcMarginProduct(512, num_class),
        'CosMargin': CosMarginProduct(512, num_class),
    }[config.LOSS.TYPE]

    # --------------------------------loss function and optimizer-----------------------------
    optimizer_sgd = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD)
    optimizer_adam = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                lr=config.TRAIN.LR)

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_sgd
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optimizer_adam
    else:
        raise ValueError('unknown optimizer type')

    if args.focal:
        criterion = FocalLoss(gamma=2).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    start_epoch = config.TRAIN.START_EPOCH
    if config.NETWORK.PRETRAINED:
        model, classifier = utils.load_pretrained(model, classifier, final_output_dir)

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, classifier = \
            utils.load_checkpoint(model, optimizer, classifier, final_output_dir)

    lr_step = [step * len(real_loader) for step in config.TRAIN.LR_STEP]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, lr_step, config.TRAIN.LR_FACTOR)

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    classifier = torch.nn.DataParallel(classifier, device_ids=gpus).cuda()

    logger.info(model)
    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    logger.info('length of train Database: ' + str(len(syn_loader.dataset)) + '  Batches: ' + str(len(syn_loader)))
    logger.info('length of real Database: ' + str(len(real_loader.dataset)))
    logger.info('Number of Identities: ' + str(num_class))

    # ----------------------------------------train----------------------------------------
    best_acc = 0.0
    best_model = False
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):

        train(syn_loader, real_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir, config, lr_scheduler)
        perf_acc, _ = lfw_eval(model, None, config, test_loader, tb_log_dir, epoch)

        if perf_acc > best_acc:
            best_acc = perf_acc
            best_model = True
        else:
            best_model = False

        logger.info('current best accuracy {:.5f}'.format(best_acc))
        logger.info('saving checkpoint to {}'.format(final_output_dir))
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'model': args.cfg,
            'state_dict': model.module.state_dict(),
            'perf': perf_acc,
            'optimizer': optimizer.state_dict(),
            'classifier': classifier.module.state_dict(),
        }, best_model, final_output_dir)
    # save best model with its acc
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    shutil.move(os.path.join(final_output_dir, 'model_best.pth.tar'), 
                os.path.join(final_output_dir, 'model_best_{}_{:.4f}.pth.tar'.format(time_str, best_acc)))

if __name__ == '__main__':
    main()
