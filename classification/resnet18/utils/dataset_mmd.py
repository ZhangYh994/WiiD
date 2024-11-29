import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, maxImgNum=1e5):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    # random.shuffle(images)
    print(len(images))
    return images[:min(maxImgNum, len(images))]

def get_index(cls_list, cls):
    for i, c in enumerate(cls_list):
        if c == cls:
            return i
    return -1

def make_labeled_dataset(dir, cls_list, maxImgNum=1e5):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    targets = []
    for root, _, fnames in sorted(os.walk(dir)):
        cls_num = get_index(cls_list, root.split('/')[-1])
        if cls_num != -1:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
                    targets.append(cls_num)
    # random.shuffle(images)
    print(len(images))
    return images[:min(maxImgNum, len(images))], targets[:min(maxImgNum, len(targets))]

class AlignedDataset(torch.utils.data.Dataset):
    def __init__(self, rootA='', rootB='', rootC='', maxImgNum = 10000, transform=None, transform2=None, transform3=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224,(0.5,1)),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transform
        if transform2 is None:
            self.transform2 = self.transform
        else:
            self.transform2 = transform2

        if transform3 is None:
            self.transform3 = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  ])
        else:
            self.transform3 = transform3

        cls_list = ['Bicycle', 'Car', 'Motorbike', 'Bus', 'Boat', 'Cat', 'Dog', 'Bottle', 'Cup', 'Chair']
        #               0        1        2          3       4      5      6        7       8        9

        # load images from '/path/to/data/trainA'
        self.A_paths, self.targets = make_labeled_dataset(rootA, cls_list, maxImgNum)
        # load images from '/path/to/data/trainB'
        self.B_paths = make_dataset(rootB, maxImgNum)

        self.C_paths = make_dataset(rootC, maxImgNum)
        self.C_index = np.arange(len(self.C_paths))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        assert self.A_size == self.B_size, 'The number of images in A and B must be equal.'

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform(A_img)
        B = self.transform2(B_img)

        small_idx = np.random.choice(self.C_index)
        C_path = self.C_paths[small_idx]
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform(C_img)

        return torch.stack([A, B], 0), self.targets[index], C

    def __len__(self):
        return self.A_size
