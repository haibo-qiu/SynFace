from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

data_root = 'data'
config= edict()

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = ''
config.NETWORK.IMAGE_SIZE = (112, 96)
config.NETWORK.WEIGHT_MODEL = ''

config.LOSS = edict()
config.LOSS.TYPE = 'ArcMargin' # 'CosMargin'
config.LOSS.WEIGHT = 1.0

# DATASET related params
config.DATASET = edict()
config.DATASET.WEBFACE_ROOT = data_root + 'datasets/casia-maxpy-clean/CASIA-Images-Align-112x96/'
config.DATASET.SYN_ROOT = data_root + 'datasets/casia-maxpy-clean/CASIA-Generated-Align-112x96/'
config.DATASET.LMDB_FILE = data_root + 'datasets/CASIA-WebFace/CASIA-112x96-LMDB/train.lmdb'
config.DATASET.TRAIN_DATASET = 'WebFace'
config.DATASET.TEST_DATASET = 'LFW'
config.DATASET.NUM_CLASS = 10567
config.DATASET.IS_GRAY = False

# LFW related
config.DATASET.LFW_PATH = data_root + 'datasets/lfw/lfw-112X96/'
# config.DATASET.LFW_PATH = data_root + 'datasets/lfw/lfw-generated-112X96/'
config.DATASET.LFW_PAIRS = data_root + 'datasets/lfw/pairs.txt'
config.DATASET.LFW_CLASS = 6000

# train
config.TRAIN = edict()
config.TRAIN.OUTPUT_DIR = os.getcwd() + '/output'
config.TRAIN.LOG_DIR = os.getcwd() + '/log'
config.TRAIN.BACKBONE_MODEL = 'LResNet50E_IR'
config.TRAIN.MODEL = 'LResNet50E_IR'
config.TRAIN.GPUS = '0,1,2,3,4,5,6,7'
config.TRAIN.WORKERS = 8
config.TRAIN.PRINT_FREQ = 100
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [10, 20]
config.TRAIN.LR = 0.1
config.TRAIN.LR_FREEZE = 0.1

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0005
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0
config.TRAIN.NUM_ID = 10000
config.TRAIN.SAMPLES_PERID = 50

config.TRAIN.START_EPOCH = 0
config.TRAIN.END_EPOCH = 30
config.TRAIN.RESUME = ''
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True
config.TRAIN.DM = 0

config.REAL = edict()
config.REAL.NUM_ID = 1000
config.REAL.SAMPLES_PERID = 10

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 32
config.TEST.SHUFFLE = False
config.TEST.WORKERS = 8
config.TEST.STATE = ''
config.TEST.MODEL_FILE = ''


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file=''):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    config_file = config.DATASET.TRAIN_DATASET + '.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{loss_type}_{lr}_{epoch}_{dm}_{num_id}k_{samples_perid}_{real_num_id}k_{real_samples_perid}_{batch_size}'.format(
        model=cfg.TRAIN.MODEL, 
        loss_type=cfg.LOSS.TYPE,
        lr=cfg.TRAIN.LR,
        epoch=cfg.TRAIN.END_EPOCH,
        dm=cfg.TRAIN.DM,
        num_id=cfg.TRAIN.NUM_ID // 1000,
        samples_perid=cfg.TRAIN.SAMPLES_PERID,
        real_num_id=cfg.REAL.NUM_ID // 1000,
        real_samples_perid=cfg.REAL.SAMPLES_PERID,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )

    full_name = '{dataset}_{name}'.format(
        dataset=os.path.basename(config.DATASET.WEBFACE_ROOT),
        name=name)

    return name, full_name


if __name__ == '__main__':
    import sys
    # gen_config(sys.argv[1])
    gen_config()
