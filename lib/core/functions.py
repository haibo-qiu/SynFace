from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import logging
import pdb
import os
import numpy as np

import torch
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def train(syn_loader, real_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir, config, lr_scheduler):
    model.train()
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (syn_data, real_data) in enumerate(zip(syn_loader, real_loader)):
        syn_img, syn_label = syn_data
        real_img, real_label = real_data

        syn_img, real_img, syn_label, real_label = syn_img.cuda(), real_img.cuda(), syn_label.cuda(), real_label.cuda()

        # mixup syn img with real img
        if config.TRAIN.DM:
            lam = random.choice(np.linspace(0, 1.05, 21, endpoint=False))
        else:
            lam = 1
        img = lam * syn_img + (1 - lam) * real_img

        feature = model(img)

        # for syn img
        label_a = syn_label[:, 0].squeeze().long()
        w_a = syn_label[:, 1].squeeze().float()
        label_b = syn_label[:, 2].squeeze().long()
        w_b = syn_label[:, 3].squeeze().float()

        # extend
        w_a = w_a.reshape((w_a.size(0), 1)).repeat(1, config.TRAIN.NUM_ID + config.REAL.NUM_ID)
        w_b = w_b.reshape((w_b.size(0), 1)).repeat(1, config.TRAIN.NUM_ID + config.REAL.NUM_ID)
        # pdb.set_trace()

        # compute output
        output = w_a * classifier(feature, label_a) + w_b * classifier(feature, label_b)
        output = lam * output + (1 - lam) * classifier(feature, real_label)

        loss_a = w_a[:, 0].squeeze() * criterion(output, label_a)
        loss_b = w_b[:, 0].squeeze() * criterion(output, label_b)

        loss = lam * (loss_a.mean() + loss_b.mean()) + (1 - lam) * criterion(output, real_label).mean()

        loss_display += loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        iters = epoch * len(syn_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:
            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Lr:{:.4e}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(syn_loader), 100. * batch_idx / len(syn_loader),
                    iters, loss_display, optimizer.param_groups[0]['lr'], time_used, speed) + INFO)
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
            time_curr = time.time()
            loss_display = 0.0

