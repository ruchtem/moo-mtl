# Adapted from https://github.com/intel-isl/MultiObjectiveOptimization/blob/master/multi_task/loaders/cityscapes_loader.py
import random
import os
import torch
import numpy as np
from PIL import Image

from torch.utils import data
from pathlib import Path

import matplotlib.pyplot as plt


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask, ins, depth):
        img, mask, ins, depth = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L'), Image.fromarray(ins, mode='I'), Image.fromarray(depth, mode='F')
        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size

        for a in self.augmentations:
            img, mask, ins, depth = a(img, mask, ins, depth)

        return np.array(img), np.array(mask, dtype=np.uint8), np.array(ins, dtype=np.uint64), np.array(depth, dtype=np.float32)


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        if _sysrand.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), ins.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, ins, depth


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        rotate_degree = _sysrand.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), ins.rotate(rotate_degree, Image.NEAREST), depth.rotate(rotate_degree, Image.NEAREST)


def resize(img, size, mode=Image.NEAREST):
    size = tuple(reversed(size))
    img = Image.fromarray(img)
    img = img.resize(size, mode)
    return np.array(img)


class CITYSCAPES(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [ 0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(20), colors))

    val_identifiers = None

    def __init__(self, split, root='data/cityscapes', dim=(3, 256, 512), ann_dim=(32, 64), val_size=0.2, **kwargs):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        assert split in ['train', 'val', 'test']
        self.root = root
        self.split = split
        self.dim = dim
        self.ann_dim = ann_dim
        self.n_classes = 19 + 1 # no classes + background

        def read_files(path, suffix=''):
            files = list(Path(path).glob(f'**/*{suffix}.png'))
            def find(s, ch):
                return [i for i, ltr in enumerate(s) if ltr == ch]
            return {i.name[:find(i.name, '_')[2]]: i for i in files}
        
        def filter(files, filter_list):
            return 

        if CITYSCAPES.val_identifiers is None and val_size > 0:
            # create the validation set
            files = list(read_files(os.path.join(self.root, 'leftImg8bit', 'train')).keys())
            random.shuffle(files)
            CITYSCAPES.val_identifiers = files[:int(len(files) * val_size)]

        # self.depth_mean = 7496.97
        # self.depth_std = 7311.58

        if split == 'train' or split == 'val':
            self.images_base = os.path.join(self.root, 'leftImg8bit', 'train')
            self.annotations_base = os.path.join(self.root, 'gtFine', 'train')
            self.depth_base = os.path.join(self.root, 'disparity',  'train')
        else:
            self.images_base = os.path.join(self.root, 'leftImg8bit', 'val')
            self.annotations_base = os.path.join(self.root, 'gtFine', 'val')
            self.depth_base = os.path.join(self.root, 'disparity',  'val')

        self.images = read_files(self.images_base)
        self.labels = read_files(self.annotations_base, suffix='gtFine_labelIds')
        self.instances = read_files(self.annotations_base, suffix='gtFine_instanceIds')
        self.depth = read_files(self.depth_base)

        if CITYSCAPES.val_identifiers is not None:
            if split == 'train':
                self.images = {k:v for k, v in self.images.items() if k not in CITYSCAPES.val_identifiers}
                self.labels = {k:v for k, v in self.labels.items() if k not in CITYSCAPES.val_identifiers}
                self.instances = {k:v for k, v in self.instances.items() if k not in CITYSCAPES.val_identifiers}
                self.depth = {k:v for k, v in self.depth.items() if k not in CITYSCAPES.val_identifiers}
            elif split == 'val':
                self.images = {k:v for k, v in self.images.items() if k in CITYSCAPES.val_identifiers}
                self.labels = {k:v for k, v in self.labels.items() if k in CITYSCAPES.val_identifiers}
                self.instances = {k:v for k, v in self.instances.items() if k in CITYSCAPES.val_identifiers}
                self.depth = {k:v for k, v in self.depth.items() if k in CITYSCAPES.val_identifiers}


        self.identifiers = list(self.images.keys())

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.no_instances =  [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 0
        self.valid_classes += [0]   # background
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if split == 'train':
            self.augmentations = Compose([
                RandomRotate(10),
                RandomHorizontallyFlip()
            ])
        else:
            self.augmentations = None

        print("Found %d images" % len(self.identifiers))


    def __len__(self):
        return len(self.identifiers)


    def __getitem__(self, index):
        i = self.identifiers[index]

        img = np.array(Image.open(self.images[i]))
        lbl = np.array(Image.open(self.labels[i]))
        ins = np.array(Image.open(self.instances[i]))
        depth = np.array(Image.open(self.depth[i]) , dtype=np.float32)

        img = resize(img, self.dim[-2:], mode=Image.BICUBIC)
        lbl = resize(lbl, self.dim[-2:])
        ins = resize(ins, self.dim[-2:])
        depth = resize(depth, self.dim[-2:])
        
        # depth[depth!=0] = (depth[depth!=0] - self.DEPTH_MEAN[depth!=0]) / self.DEPTH_STD
        if self.augmentations is not None:
            img, lbl, ins, depth = self.augmentations(np.array(img, dtype=np.uint8), np.array(lbl, dtype=np.uint8), np.array(ins, dtype=np.int32), np.array(depth, dtype=np.float32))

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        ins_y, ins_x = self.encode_instancemap(lbl, ins)

        classes = np.unique(lbl)

        if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
            # print('after det', classes,  np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img, lbl, ins, depth = self.transform(img, lbl, ins_y, ins_x, depth)

        return {
            'data': img,
            'labels_segm': lbl,
            'labels_inst': ins,
            'labels_depth': depth.unsqueeze(0),
        }


    def transform(self, img, lbl, ins_y, ins_x, depth):

        img = resize(img, self.dim[-2:], mode=Image.BICUBIC)
        img = img.transpose(2, 0, 1)
        img = img / 256

        # classes = np.unique(lbl)

        lbl = resize(lbl, self.ann_dim)
        ins_y = resize(ins_y, self.ann_dim)
        ins_x = resize(ins_x, self.ann_dim)
        depth = resize(depth, self.ann_dim)

        # if not np.all(classes == np.unique(lbl)):
        #     print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
            print('after det', np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        ins = np.stack((ins_y, ins_x))
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        ins = torch.from_numpy(ins).float()
        depth = torch.from_numpy(depth).float()
        return img, lbl, ins, depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask==_validc] = self.class_map[_validc]
        return mask

    def encode_instancemap(self, mask, ins):
        ins[mask==self.ignore_index] = self.ignore_index
        for _no_instance in self.no_instances:
            ins[ins==_no_instance] = self.ignore_index
        ins[ins==0] = self.ignore_index

        instance_ids = np.unique(ins)
        sh = ins.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]),np.arange(sh[1]), indexing='ij')

        out_ymap, out_xmap = np.meshgrid(np.arange(sh[0]),np.arange(sh[1]), indexing='ij')
        out_ymap = np.ones(ymap.shape)*self.ignore_index
        out_xmap = np.ones(xmap.shape)*self.ignore_index

        for instance_id in instance_ids:
            if instance_id == self.ignore_index:
                continue
            instance_indicator = (ins == instance_id)
            coordinate_y, coordinate_x = np.mean(ymap[instance_indicator]), np.mean(xmap[instance_indicator])
            out_ymap[instance_indicator] = ymap[instance_indicator] - coordinate_y
            out_xmap[instance_indicator] = xmap[instance_indicator] - coordinate_x

        return out_ymap, out_xmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dst = CITYSCAPES('train')
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs = data['data']
        labels = data['labels_segm']
        instances = data['labels_inst']
        depth = data['labels_depth']

        f, axarr = plt.subplots(bs,5)
        for j in range(bs):
            img = imgs[j].numpy()
            img = np.moveaxis(img, 0, -1)
            axarr[j][0].imshow(img)
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            axarr[j][2].imshow(instances[j,0,:,:])
            axarr[j][3].imshow(instances[j,1,:,:])
            d = depth[j]
            axarr[j][4].imshow(d.squeeze())
        plt.show()
        plt.close()