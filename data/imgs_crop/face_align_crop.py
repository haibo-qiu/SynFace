import os
import argparse

import logging
import torch
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from dataset import Dataset_Folder
from detector import detect_faces
from get_nets import PNet, RNet, ONet
from align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description = "face alignment")
parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "./data/test", type = str)
parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "./data/test_Aligned", type = str)
parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
parser.add_argument("-j", "--procs", help = "number of procs per gpu", default = 4, type = int)
parser.add_argument("--dist-url", help = "init_method for distributed training", default = 'tcp://127.0.0.1:8081', type = str)
parser.add_argument("--dist-backend", help = "distributed backend", default = 'nccl', type = str)
args = parser.parse_args()

def mp_print(msg, i):
    if dist.get_rank() == 0 and i % 400 == 0:
        print(msg)

def make_dirs(source_root, dest_root):
    for subfolder in os.listdir(source_root):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.makedirs(os.path.join(dest_root, subfolder))

def main():
    global args
    print(args)
    # to use mp, dirs need to be created first
    make_dirs(args.source_root, args.dest_root)
    print('Done making new dirs')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    nprocs = args.procs * torch.cuda.device_count()

    torch.multiprocessing.spawn(test, args=(args, nprocs), nprocs=nprocs, join=True, daemon=False)

def test(rank, args, world_size):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = False) * scale

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    # LOAD MODELS
    pnet = PNet().cuda()
    rnet = RNet().cuda()
    onet = ONet().cuda()

    data = Dataset_Folder(source_root)
    start_index, end_index = 0, len(data)

    for i in range(start_index+dist.get_rank(), end_index, dist.get_world_size()):
        img_path = str(data.imgs[i]) 
        try:
            img = Image.open(img_path)
        except Exception:
            print("process[{}]: {} is discarded due to read exception!".format(dist.get_rank(), img_path))
            continue

        # logger.info("process[{}]: dealing with {}".format(dist.get_rank(), img_path))
        mp_print("process[{}]: {}/{}".format(dist.get_rank(), i // dist.get_world_size(), len(range(start_index+dist.get_rank(), end_index, dist.get_world_size()))), i)
        try: # Handle exception
            _, landmarks = detect_faces(img, nets=[pnet, rnet, onet])
        except Exception:
            print("process[{}]: {} is discarded due to exception!".format(dist.get_rank(), img_path))
            continue
        if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
            print("process[{}]: {} is discarded due to non-detected landmarks!".format(dist.get_rank(), img_path))
            # raise ValueError('error!!!')
            continue

        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference)
        img_warped = Image.fromarray(warped_face)

        file_names = img_path.split('/')
        img_name = '/'.join(file_names[-2:])
        dest_path = os.path.join(dest_root, img_name)
        img_warped.save(dest_path)

if __name__ == '__main__':
    main()
