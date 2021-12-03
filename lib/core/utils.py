from __future__ import print_function
from __future__ import division
import numpy as np
import os
import time
import torch
import random
import socket
import logging
from pathlib import Path
from datetime import datetime
from lib.core.config import get_model_name

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) 
    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) 
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# ############## logger related ######################
def create_temp_logger():
    output_dir = 'temp'
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir

    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = root_output_dir / 'tensorboard'
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def create_logger(cfg, cfg_name, dataset, phase='train'):
    root_output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0] 
    dataset = dataset.split('/')[-2]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file

    fmt = "[%(asctime)s] %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # add file handler to save the log to file
    fh = logging.FileHandler(final_log_file)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # add console handler to output log on screen
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.propagate = False

    tensorboard_file = 'tensorboard_{}'.format(time_str)
    tensorboard_log_dir = final_output_dir / tensorboard_file
    print('creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

# ############## logger related ######################


# ############## dir related ######################
def checkdir(path):
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ############## dir related ######################

# ################ network training #############################
def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()

def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        optimizer.param_groups[0]['lr'] = args.lr_freeze * lr 
        optimizer.param_groups[1]['lr'] = lr
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr
    else:
        pass

# ################ network training #############################

# ################ save and load #############################
def load_pretrained(model, classifier, output_dir, filename='model_baseline_9443.pth.tar'):
    file = os.path.join(output_dir, filename)
    file = file if os.path.isfile(file) else os.path.join('model', filename)
    if os.path.isfile(file):
        print('load pretrained model from {}'.format(file))
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        classifier.load_state_dict(checkpoint['classifier'])

        return model, classifier

    else:
        print('no checkpoint found at {}'.format(file))
        return model, classifier

def load_checkpoint(model, optimizer, classifier, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        classifier.load_state_dict(checkpoint['classifier'])
        print('load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, classifier

    else:
        print('no checkpoint found at {}'.format(file))
        return 0, model, optimizer, classifier

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if is_best and 'state_dict' in states:
        state_info = {'state_dict': states['state_dict'],
                      'classifier': states['classifier']}
        # torch.save(state_info, os.path.join(output_dir, 'model_best_{}_{:.4f}.pth.tar'.format(time_str, states['perf'])))
        torch.save(state_info, os.path.join(output_dir, 'model_best.pth.tar'))

# ################ save and load #############################

############### mixup related #################
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

############### mixup related #################
