"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data

from datasets.data_utils import CDDataAugmentation


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))

    
class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img_weak, img_B_weak], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)
        [img_strong, img_B_strong] = self.augm.transform_strong([img_weak, img_B_weak],[])
        [img_strong2, img_B_strong2] = self.augm.transform_strong2([img_weak, img_B_weak],[])
        [img_strong3, img_B_strong3] = self.augm.transform_strong3([img_weak, img_B_weak],[])
        
        img_aug = torch.stack([img_strong3, img_B_strong3, img_weak,img_B_weak,img_strong,img_B_strong,img_strong2,img_B_strong2],dim=1)

        return {'A': img_weak, 'B': img_B_weak, 'name': name, 'aug': img_aug}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        [img_weak, img_B_weak], [label] = self.augm.transform([img, img_B],[label], to_tensor=self.to_tensor)
        [img_strong, img_B_strong] = self.augm.transform_strong([img_weak, img_B_weak],[])
        [img_strong2, img_B_strong2] = self.augm.transform_strong2([img_weak, img_B_weak],[])
        [img_strong3, img_B_strong3] = self.augm.transform_strong3([img_weak, img_B_weak],[])
        
        if np.count_nonzero(label)==0: class_label = 0 
        else: class_label = 1
        
        img_aug = torch.stack([img_strong3, img_B_strong3, img_weak,img_B_weak, img_strong, img_B_strong, img_strong2,img_B_strong2],dim=1)
        
        return {'name': name, 'A': img_weak, 'B': img_B_weak, 'L': label, 'cls':class_label, 'aug': img_aug}

