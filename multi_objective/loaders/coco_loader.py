import os
import torch
import random
import numpy as np
import torchvision.transforms as t
from torchvision.datasets import CocoDetection
from PIL import Image
from fvcore import transforms


class ResizeShortestEdge():

    def __init__(self, target_size, interp='bilinear') -> None:
        self.target_size = target_size
        self.interp = interp
    

    def get_transform(self, image):
        h, w = image.shape[:2]
        max_w, max_h = self.target_size

        if w <= max_w and h <= max_h:
            new_w = w
            new_h = h
        else:
            scale_w = max_w / w
            scale_h = max_h / h
            scale = min(scale_w, scale_h)
            assert scale < 1.

            new_w = int(w * scale)
            new_h = int(h * scale)
        
        return transforms.ScaleTransform(h, w, new_h, new_w, self.interp)
        

class RandomFlip():

    def __init__(self, p=0.5, horizontal=True, vertical=False) -> None:
        self.p = p
        self.horizontal = horizontal
        self.vertical = vertical
    

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = random.random() < self.p
        if do:
            if self.horizontal:
                return transforms.HFlipTransform(w)
            elif self.vertical:
                return transforms.VFlipTransform(h)
        else:
            return transforms.NoOpTransform()


class RandomRotation():

    def __init__(self, angle) -> None:
        self.angle = angle
    

    def get_transform(self, image):
        h, w = image.shape[:2]
        angle = random.uniform(self.angle[0], self.angle[1])
        if angle % 360 == 0:
            return transforms.NoOpTransform()
        return RotationTransform(h, w, angle)

# taken from https://github.com/facebookresearch/detectron2/blob/2455e4790f470bba54299c049410fc0713ae7529/detectron2/data/transforms/transform.py#L162 and adaped
import cv2
class RotationTransform():
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self.h = h
        self.w = w
        self.angle = angle
        self.expand = expand
        self.image_center = image_center
        self.center = center
        self.interp = interp
        self.bound_w = bound_w
        self.bound_h = bound_h
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation
    
    def apply_box(self, box):
        box1 = self.apply_coords(box[:, 0:2])
        box2 = self.apply_coords(box[:, 2:4])
        return np.hstack((box1, box2))

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class COCO(CocoDetection):

    def __init__(self, split, root='data/coco', dim=(3, 256, 256)) -> None:
        assert split in ['train', 'val', 'test']
        if split=='train' or split=='val':
            self.annFile = os.path.join(root, 'annotations', 'instances_train2017.json')
            self.img_dir = os.path.join(root, 'train2017')
            
        else:
            self.annFile = os.path.join(root, 'annotations', 'instances_val2017.json')
            self.img_dir = os.path.join(root, 'val2017')

        super().__init__(self.img_dir, self.annFile)

        self.min_area = 50.

        self.transforms = [
            ResizeShortestEdge(target_size=(512, 512)),
        ]

        self.train_transforms = [
            RandomFlip(),
            RandomRotation((-10, 10)),
        ]


    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.img_dir, path)).convert("RGB")

    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        image = np.array(image)

        anns = self.coco.getAnnIds(id, areaRng=[self.min_area, float('inf')], iscrowd=False)
        anns = self.coco.loadAnns(anns)

        masks = np.stack([self.coco.annToMask(a) for a in anns], axis=-1)
        bboxs = np.stack([a['bbox'] for a in anns])     # bbox_x, bbox_y, bbox_w, bbox_h
        cats = np.stack([a['category_id'] for a in anns])

        # convert boxes from xywh to xyxy
        bboxs[:, 2] += bboxs[:, 0]
        bboxs[:, 3] += bboxs[:, 1]

        # (height, width, channels)
        for transform in self.transforms:
            t = transform.get_transform(image)
            image = t.apply_image(image)
            masks = t.apply_segmentation(masks)
            bboxs = t.apply_box(bboxs)
        
        for transform in self.train_transforms:
            t = transform.get_transform(image)
            image = t.apply_image(image)
            masks = t.apply_segmentation(masks)
            bboxs = t.apply_box(bboxs)
        
        bboxs[:, 2] -= bboxs[:, 0]
        bboxs[:, 3] -= bboxs[:, 1]

        image = torch.from_numpy(image.copy())
        masks = torch.from_numpy(masks.copy())
        bboxs = torch.from_numpy(bboxs)
        cats = torch.from_numpy(cats)

        # (channels, height, width)
        image = torch.movedim(image, -1, 0)
        masks = torch.movedim(masks, -1, 0)

        return {
            'data': image,
            'masks': masks,
            'bboxs': bboxs,
            'cats': cats,
        }


if __name__ == '__main__':
    from torch.utils import data
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = [[128,  64, 128],
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

    def collate_fn(x):
        return x
    
    def mask_to_colors(cats, masks):
        h, w = masks.shape[-2:]
        masks = masks.numpy().astype(np.bool)
        if masks.ndim == 2:
            masks = np.expand_dims(masks, 0)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c, m in zip(cats, masks):
            col = colors[c % len(colors)]
            img[m] = col
        return img

    dataset = COCO('test')
    bs=2
    loader = data.DataLoader(dataset, bs, num_workers=0, collate_fn=collate_fn)

    for b in loader:
        f, axarr = plt.subplots(bs, 2)
        for j, inst in enumerate(b):
            img = inst['data']
            masks = inst['masks']
            bboxs = inst['bboxs']
            cats = inst['cats']
            axarr[j][0].imshow(torch.movedim(img, 0, -1))
            for box in bboxs:
                bbox_x, bbox_y, bbox_w, bbox_h = box
                rect = Rectangle((bbox_x, bbox_y),bbox_w, bbox_h,linewidth=1,edgecolor='r',facecolor='none')
                axarr[j][0].add_patch(rect)
            axarr[j][1].imshow(mask_to_colors(cats, masks))
        plt.show()
        print()


    print(dataset)