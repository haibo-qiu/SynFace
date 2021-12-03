import time
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

cudnn.benchmark = True

logger = logging.getLogger(__name__)

def extractDeepFeature(img, model, is_gray):
    img = img.to('cuda')
    fc = model(img)
    fc = fc.to('cpu').squeeze()
    
    return fc

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def cosine_similarity(f1, f2):
    # compute cosine_similarity for 2-D array
    f1 = f1.numpy()
    f2 = f2.numpy()

    A = np.sum(f1*f2, axis=1)
    B = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-5

    return A / B

def compute_distance(img1, img2, model,flag, is_gray):
        f1 = extractDeepFeature(img1, model, is_gray)
        f2 = extractDeepFeature(img2, model, is_gray)

        distance = cosine_similarity(f1, f2)

        flag = flag.squeeze().numpy()
        return np.stack((distance, flag), axis=1)

def obtain_acc(predicts, num_class, start):
    accuracy = []
    thd = []
    folds = KFold(n=num_class, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    end = time.time()
    time_used = (end - start) / 60.0
    logger.info('LFW_ACC={:.4f} std={:.4f} thd={:.4f} time_used={:.4f} mins'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd), time_used))
    # print('LFW_ACC={:.4f} std={:.4f} thd={:.4f} time_used={:.4f} mins'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd), time_used))
    return np.mean(accuracy)


def eval(model, model_path, config, test_loader, tb_log_dir, epoch, is_gray=False):
    if model_path:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

    predicts = np.zeros(shape=(len(test_loader.dataset), 2))

    model.eval()
    start = time.time()

    cur = 0
    with torch.no_grad():
        for batch_idx, (img1, img2, flag) in enumerate(test_loader):
            predicts[cur:cur+flag.shape[0]] = compute_distance(img1, img2, model, flag, is_gray)
            cur += flag.shape[0]
    assert cur == predicts.shape[0]

    accuracy = obtain_acc(predicts, config.DATASET.LFW_CLASS, start)

    # visualize the masks stats
    writer = SummaryWriter(tb_log_dir)
    writer.add_scalar('LFW_ACC', np.mean(accuracy), epoch)
    writer.close()

    return np.mean(accuracy), predicts

