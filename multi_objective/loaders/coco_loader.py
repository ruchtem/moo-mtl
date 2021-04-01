import os
import torch
import numpy as np
import torchvision.transforms as t
from torchvision.datasets import CocoDetection
from PIL import Image
from fvcore import transforms



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


    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.img_dir, path)).convert("RGB")

    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        image = np.array(image)

        anns = self.coco.getAnnIds(id, areaRng=[self.min_area, float('inf')], iscrowd=False)
        anns = self.coco.loadAnns(anns)

        masks = np.stack([self.coco.annToMask(a) for a in anns])
        bboxs = np.stack([a['bbox'] for a in anns])
        cats = np.stack([a['category_id'] for a in anns])

        # TODO: implement transforms from fvcore

        image = torch.from_numpy(image)
        masks = torch.from_numpy(masks)
        bboxs = torch.from_numpy(bboxs)
        cats = torch.from_numpy(cats)

        return {
            'data': image,
            'masks': masks,
            'bboxs': bboxs,
            'cats': cats,
        }


if __name__ == '__main__':
    from torch.utils import data

    def collate_fn(x):
        #x['data'] = torch.stack([i['data'] for i in x])
        return x

    dataset = COCO('test')
    loader = data.DataLoader(dataset, 4, num_workers=0, collate_fn=collate_fn)

    for b in loader:
        print(b)
        break


    print(dataset)