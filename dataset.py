import PIL
import json
import torch
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
import os
import numpy as np

from voc_utils import get_all_obj_and_box

import random


class DetectionDataset(Dataset):

    def _load(self, i):
        filename = self.filenames[i]
        boxes = self.boxes[i]
        
        x = default_loader(filename)
        
        da_params = self.data_augmentation_params
        if da_params is None:
            da_params = {}
        u = random.uniform(0, 1)
        if u <= da_params.get('patch_proba', 0):
            min_scale = da_params.get('min_scale', 0.1)
            max_scale = da_params.get('max_scale', 1)
            min_ar = da_params.get('min_aspect_ratio', 0.5)
            max_ar = da_params.get('max_aspect_ratio', 2)
            max_nb_trials = da_params.get('nb_trials', 50)
            scale = random.uniform(min_scale, max_scale)
            ar = random.uniform(min_ar, max_ar)
            boxes_ = []
            nb_trials = 0
            while len(boxes_) == 0 and nb_trials < max_nb_trials:
                x_, crop_box = _random_patch(self.rng, x, scale, ar)
                boxes_ = [(box, cat) for box, cat in boxes if box_in_box(box, crop_box)]
                bx, by, bw, bh = crop_box
                boxes_ = [((x - bx, y - by, w, h), cat) for (x, y, w, h), cat in boxes_]
                nb_trials += 1
            if len(boxes_) > 0:
                boxes = boxes_
                x = x_
        assert len(boxes) > 0, i
        # flip
        u = random.uniform(0, 1)
        if u <= da_params.get('flip_proba', 0):
            x = x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            boxes = [((x.size[0] - bx - bw, by, bw, bh), cat) for (bx, by, bw, bh), cat in boxes] 
        orig_w, orig_h = x.size
        x = self.transform(x)
        sw = 1 / orig_w
        sh = 1 / orig_h
        boxes = [((x * sw, y * sh, w * sw, h * sh), cat) for ((x, y, w, h), cat) in boxes]
        m = _build_mask(x, boxes)
        m = m.view(1, m.size(0), m.size(1))
        def sort_box(b):
            box, cat = b
            x, y, w, h = box
            return (y+h/2, x+w/2)#sory by y, then by x
        boxes = sorted(boxes, key=sort_box)
        return x, m, boxes

    def __getitem__(self, i):
        return self._load(i)

    def __len__(self):
        return len(self.boxes)


class COCO(DetectionDataset):
    def __init__(self, folder='data/coco', 
                 split='train2014', 
                 image_size=300,
                 data_augmentation_params=None,
                 classes=None, transform=None, random_state=42):
        self.image_size = image_size
        self.folder = folder
        self.annotations_folder = os.path.join(folder, 'annotations')
        self.split = split
        self.transform = transform
        self.classes = classes
        self.data_augmentation_params = data_augmentation_params
        self.rng = np.random.RandomState(random_state)
        self._load_annotations()
    
    def _load_annotations(self):
        A = json.load(open(os.path.join(self.annotations_folder, 'instances_{}.json'.format(self.split))))
        B = json.load(open(os.path.join(self.annotations_folder, 'captions_{}.json'.format(self.split))))
        class_id_name = {a['id']: a['name'] for a in A['categories']}
        image_id_to_filename = {}
        for b in B['images']:
            image_id_to_filename[b['id']] = b['file_name']
        keys = list(image_id_to_filename.keys())
        keys = sorted(keys)
        self.rng.shuffle(keys)
        index_to_filename = {i: image_id_to_filename[k] for i, k in enumerate(keys)}
        image_id_to_index = {k: i for i, k in enumerate(keys)}
        
        if self.classes:
            classes = set(self.classes)
        else:
            classes = set()
            for a in A['annotations']:
                cat = class_id_name[a['category_id']]
                classes.add(cat)
        self.classes = sorted(list(classes))
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        B = defaultdict(list)
        for a in A['annotations']:
            bbox = a['bbox']
            cat = class_id_name[a['category_id']]
            if cat in classes:
                cat_id = self.class_to_idx[cat]
                B[image_id_to_index[a['image_id']]].append((bbox, cat_id))
        indexes = list(index_to_filename.keys())
        self.boxes = [B[ind] for ind in indexes if len(B[ind]) > 0]
        self.filenames = [index_to_filename[ind] for ind in indexes if len(B[ind]) > 0]
        self.filenames = [os.path.join(self.folder, self.split,  f) for f in self.filenames]


class VOC(DetectionDataset):
    def __init__(self, folder='data/voc', 
                 which='VOC2007', split='train',
                 image_size=300,
                 data_augmentation_params=None, 
                 classes=None, transform=None, 
                 random_state=42):
        self.folder = folder # root folder, should contain VOC2007 and/or VOC2012
        self.which = which
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.classes = classes
        self.data_augmentation_params = data_augmentation_params
        self.rng = np.random.RandomState(random_state)
        self._load_annotations()

    def _load_annotations(self):
        voc2007 = os.path.join(self.folder, 'VOC2007')
        voc2012 = os.path.join(self.folder, 'VOC2012')
        if self.which == 'VOC2007':
            paths = [voc2007]
        elif self.which == 'VOC2012':
            paths = [voc2012]
        elif self.which == 'VOC0712':
            paths = [voc2007, voc2012]
        else:
            raise ValueError('which should be voc2007 or voc2012 or voc0712')
        anns = []
        for path in paths:
            anns += get_all_obj_and_box(self.split, path, classes=self.classes)
        classes = set()
        for fname, bboxes in anns:
            for (x, y, w, h), class_name in bboxes:
                classes.add(class_name)
        classes = list(classes)
        if self.classes:
            assert set(classes) == set(self.classes)
        anns = sorted(anns)
        anns = [(fname, bboxes) for fname, bboxes in anns if len(bboxes) > 0]
        self.rng.shuffle(anns)
        self.filenames = [fname for fname, bboxes in anns]
        self.classes = sorted(classes)
        self.class_to_idx = {cl: i + 1 for i, cl in enumerate(self.classes)}
        self.class_to_idx['background'] = 0
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.boxes = [[(b, self.class_to_idx[cl]) for (b, cl)  in bboxes] for fname, bboxes in anns]

class SubSample:

    def __init__(self, dataset, nb):
        nb = min(len(dataset), nb)
        self.dataset = dataset
        self.nb = nb
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.idx_to_class
        self.transform = dataset.transform

    def __getitem__(self, i):
        return self.dataset[i]
        
    def __len__(self):
        return self.nb

def _build_mask(im, bboxes):
    m = torch.zeros(im.size(1), im.size(2)).float()
    for (x, y, w, h), cat in bboxes:
        x = int(x * m.size(1))
        y = int(y * m.size(0))
        w = int(w * m.size(1))
        h = int(h * m.size(0))
        w = max(w, 1)
        h = max(h, 1)
        xmin = min(x, m.size(1) - w)
        xmax = x + w 
        ymin = min(y, m.size(0) - h)
        ymax = y + h
        #val = 1 / (w*h)
        val = 1
        m[ymin:ymax, xmin:xmax] = val
    return m

def _random_patch(rng, im, scale, aspect_ratio):
    w, h = im.size
    wcrop = int(scale * w)
    hcrop = min(int(wcrop / aspect_ratio), h) 
    xmin, ymin = rng.randint(0, w - wcrop + 1), rng.randint(0, h - hcrop + 1)
    xmax = xmin + wcrop
    ymax = ymin + hcrop
    return im.crop((xmin, ymin, xmax, ymax)), (xmin, ymin, wcrop, hcrop)

def box_in_box(box_small, box_big):
    bsx, bsy, bsw, bsh = box_small
    bbx, bby, bbw, bbh = box_big
    if not (bsx >= bbx and bsx + bsw <= bbx + bbw):
        return False
    if not (bsy >= bby and bsy + bsh <= bby + bbh):
        return False
    return True
