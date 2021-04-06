import os
import torch
import numpy as np
import torchvision.transforms as t
from torchvision.datasets import CocoDetection
from PIL import Image
from fvcore import transforms


class ResizeLongestSide():
    
    @staticmethod
    def _cal_new_size(img_size, target_size):
        h, w = img_size
        max_w, max_h = target_size

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
        return w, h, new_w, new_h
    

    def apply_image(self, img, target_size):
        w, h, new_w, new_h = self._cal_new_size(img.shape[:2], target_size)
        t = transforms.ScaleTransform(h, w, new_h, new_w)
        return t.apply_image(img, interp='bilinear')
    
    def apply_segmentation(self, segmentation, target_size):
        w, h, new_w, new_h = self._cal_new_size(segmentation.shape[:2], target_size)
        t = transforms.ScaleTransform(h, w, new_h, new_w)
        return t.apply_segmentation(segmentation)
    
    def apply_box(self, box, img_shape, target_size):
        w, h, new_w, new_h = self._cal_new_size(img_shape, target_size)
        t = transforms.ScaleTransform(h, w, new_h, new_w)

        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]

        box = t.apply_box(box)

        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]

        return box
        



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
        self.smallest_side = 512

        self.transforms = [
            ResizeLongestSide()
        ]

        self.train_transforms = [

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

        # (height, width, channels)
        original_shape = image.shape
        for t in self.transforms:
            image = t.apply_image(image, target_size=(self.smallest_side, self.smallest_side))
            masks = t.apply_segmentation(masks, target_size=(self.smallest_side, self.smallest_side))
            bboxs = t.apply_box(bboxs, original_shape[:2], target_size=(self.smallest_side, self.smallest_side))

        image = torch.from_numpy(image)
        masks = torch.from_numpy(masks)
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
        #x['data'] = torch.stack([i['data'] for i in x])
        return x
    
    def mask_to_colors(cats, masks):
        h, w = masks.shape[-2:]
        masks = masks.numpy().astype(np.bool)
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