from pathlib import Path

class Dataset_Folder(object):
    def __init__(self, root):
        self.root = root
        self.imgs = self.get_imgs()

    def get_imgs(self):
        path = Path(self.root)
        imgs = sorted(list(path.glob('**/*.jpg')))
        if len(imgs) == 0:
            imgs = sorted(list(path.glob('**/*.png')))
        return imgs

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root='/apdcephfs/share_1290939/haiboqiu/datasets/casia-maxpy-clean/CASIA-Generated-10k-100-MixID/'
    data = Dataset_Folder(root)
    imgs = data.imgs
    print(imgs[:10])
    print(len(imgs))
        
