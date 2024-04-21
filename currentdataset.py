import numpy as np
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class CurrentDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        # Masks
        masks = []
        for i in range(num_objs):
            if i >= len(coco_annotation):
                break
            try:
                masks.append(coco.annToMask(coco_annotation[i]))
            except:
                # remove the image from current lists because of the error
                num_objs -= 1
                boxes.pop(i)
                coco_annotation.pop(i)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # # Tensorise img_id
        # img_id = torch.tensor([img_id])
        # # Size of bbox (Rectangular)
        # areas = []
        # for i in range(num_objs):
        #     areas.append(coco_annotation[i]['area'])
        # areas = torch.as_tensor(areas, dtype=torch.float32)
        # # Iscrowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        # my_annotation["image_id"] = img_id
        # my_annotation["area"] = areas
        # my_annotation["iscrowd"] = iscrowd
        my_annotation["masks"] = masks

        if self.transforms is not None:
            img = self.transforms(img)

        return torch.from_numpy(np.array(img)), my_annotation

    def __len__(self):
        return len(self.ids)
