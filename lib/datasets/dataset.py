import os
import re
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Img_Folder_Num(Dataset):
    def __init__(self, root, refer_classes, refer_length, albu_transform=None, transform=None, num_id=1000, samples_perid=10):
        super(Img_Folder_Num, self).__init__()
        self.root = root
        self.refer_length = refer_length
        self.db, self.class_to_label = self.load_db(refer_classes, refer_length, num_id, samples_perid)
        self.classes = list(self.class_to_label.keys())
        self.albu_transform = albu_transform
        self.transform = transform


    def load_db(self, refer_classes, refer_length, num_id, samples_perid):
        db = []
        class_to_label = {}
        classes = sorted(os.listdir(self.root))
        # np.random.shuffle(classes)

        assert num_id <= len(classes)
        classes = classes[:num_id]
        # classes = sorted(random.sample(classes, k=num_id)) 

        for i, class_name in enumerate(classes):
            if class_name not in class_to_label:
                class_to_label[class_name] = i + refer_classes

            img_names = os.listdir(os.path.join(self.root, class_name))
            if samples_perid <= len(img_names):
                img_names = img_names[:samples_perid]

            for img_name in img_names:
                datum = [os.path.join(class_name, img_name), i + refer_classes]
                db.append(datum)

        k = refer_length // len(db)
        db *= k
        if refer_length - len(db) > 0:
            samples = random.choices(db, k=refer_length-len(db))
            db += samples

        assert len(db) == refer_length

        return db, class_to_label


    def PIL_loader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def __getitem__(self, index):
        datum = self.db[index]
        img_path, label = datum

        img = self.PIL_loader(os.path.join(self.root, img_path))

        if self.albu_transform is not None:
            img_np = np.array(img)
            augmented = self.albu_transform(image=img_np)
            img = Image.fromarray(augmented['image'])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.db)


class Img_Folder_Mix(Dataset):
    def __init__(self, root, albu_transform=None, transform=None, num_id=10000, samples_perid=50):
        super(Img_Folder_Mix, self).__init__()
        self.root = root
        self.db, self.class_to_label = self.load_db(num_id, samples_perid)
        self.classes = list(self.class_to_label.keys())
        self.albu_transform = albu_transform
        self.transform = transform

    def load_db(self, num_id, samples_perid):
        db = []
        class_to_label = {}
        classes = sorted(os.listdir(self.root))
        if num_id < len(classes):
            classes = classes[:num_id]

        for i, class_name in enumerate(classes):
            if class_name not in class_to_label:
                class_to_label[class_name] = i

            img_names = os.listdir(os.path.join(self.root, class_name))
            if samples_perid < len(img_names):
                img_names = img_names[:samples_perid]

            for img_name in img_names:
                datum = [os.path.join(class_name, img_name), i]
                db.append(datum)

        return db, class_to_label

    def PIL_loader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def get_label(self, path):
        path = os.path.basename(path)
        template = r'(.*)_(.*)x(.*)_\((.*)\)x(.*).jpg'
        info = re.match(template, path)
        l_1, w_1 = int(info.group(2)), float(info.group(3))
        l_2, w_2 = int(info.group(4)), float(info.group(5))
        label = torch.tensor((l_1-1, w_1, l_2-1, w_2)).float()
        return label

    def __getitem__(self, index):
        datum = self.db[index]
        img_path, label = datum

        img = self.PIL_loader(os.path.join(self.root, img_path))
        label = self.get_label(img_path)

        if self.albu_transform is not None:
            img_np = np.array(img)
            augmented = self.albu_transform(image=img_np)
            img = Image.fromarray(augmented['image'])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.db)

class Img_Folder(Dataset):
    def __init__(self, root, refer_length, sync=False, albu_transform=None, transform=None):
        super(Img_Folder, self).__init__()
        self.root = root
        self.sync = sync
        self.refer_length = refer_length
        self.db, self.class_to_label = self.load_db()
        self.classes = list(self.class_to_label.keys())
        self.albu_transform = albu_transform
        self.transform = transform

    def load_db(self):
        db = []
        class_to_label = {}
        for i, class_name in enumerate(os.listdir(self.root)):
            if class_name not in class_to_label:
                class_to_label[class_name] = i

            for img_name in (os.listdir(os.path.join(self.root, class_name))):
                datum = [os.path.join(class_name, img_name), i]
                db.append(datum)

        # align the real to syn dataset length
        length = len(db)
        if length < self.refer_length:
            over_samples = random.choices(db, k=self.refer_length - length)
            db += over_samples

        elif length > self.refer_length:
            db = random.choices(db, k=self.refer_length)

        assert len(db) == self.refer_length

        return db, class_to_label

    def PIL_loader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def __getitem__(self, index):
        datum = self.db[index]
        img_path, label = datum

        img = self.PIL_loader(os.path.join(self.root, img_path))

        if self.albu_transform is not None:
            img_np = np.array(img)
            augmented = self.albu_transform(image=img_np)
            img = Image.fromarray(augmented['image'])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.db)

class LFW_Image(Dataset):
    def __init__(self, lfw_path, lfw_pairs, transform=None):
        super(LFW_Image, self).__init__()
        self.lfw_path = lfw_path
        self.pairs = self.get_pairs_lines(lfw_pairs)

        self.transform = transform

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def __getitem__(self, index):
        p = self.pairs[index].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        with open(self.lfw_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')

        with open(self.lfw_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, sameflag

    def __len__(self):
        return len(self.pairs)

