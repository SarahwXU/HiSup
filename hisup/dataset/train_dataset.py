import cv2
import random
import os.path as osp
import numpy as np

from skimage import io
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, rotate_f=None):
        self.root = root

        self.coco = COCO(ann_file)
        images_id = self.coco.getImgIds()
        self.images=images_id.copy()
        self.num_samples = len(self.images)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
        # basic information
        img_id = self.images[idx_]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_coco = self.coco.loadAnns(ids=ann_ids)
        
        ann = {
            'junctions': [],
            'juncs_index': [],
            'juncs_tag': [],
            'edges_positive': [],
            'bbox': [],
            'width': width,
            'height': height,
        }

        pid = 0
        instance_id = 0
        seg_mask = np.zeros([width, height])
        for ann_per_ins in ann_coco:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2) # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points = segm[:-1]
                junc_tags = np.ones(points.shape[0])
                if i == 0:  # outline
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2    # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            idxs = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1,0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask = np.clip(seg_mask, 0, 1)

        # load image
        image = io.imread(osp.join(self.root, file_name)).astype(float)[:, :, :3]
        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.long],
                           ['juncs_tag', np.long],
                           ['juncs_index', np.long],
                           ['bbox', np.float32],
                           ):
            ann[key] = np.array(ann[key], dtype=_type)

        # augmentation
        if self.rotate_f:
            reminder = random.randint(0, 5)
        else:
            reminder = random.randint(0, 3)
        ann['reminder'] = reminder

        if len(ann['junctions']) > 0:
            if reminder == 1:   # horizontal flip
                image = image[:, ::-1, :]
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['bbox'] = ann['bbox'][:, [2, 1, 0, 3]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                seg_mask = np.fliplr(seg_mask)
            elif reminder == 2: # vertical flip
                image = image[::-1, :, :]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [0, 3, 2, 1]]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
                seg_mask = np.flipud(seg_mask)
            elif reminder == 3: # horizontal and vertical flip
                image = image[::-1, ::-1, :]
                seg_mask = np.fliplr(seg_mask)
                seg_mask = np.flipud(seg_mask)
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [2, 3, 0, 1]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
            elif reminder == 4: # rotate 90 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width/2), (height/2)), 90, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            elif reminder == 5: # rotate 270 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            else:
                pass
            ann['mask'] = seg_mask
        else:
            ann['mask'] = np.zeros((height, width), dtype=np.float64)
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0,0,0,0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
