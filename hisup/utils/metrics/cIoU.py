"""
The code for C-IoU, adopted from https://github.com/zorzi-s/PolyWorldPretrainedNetwork/blob/main/coco_IoU_cIoU.py.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
DATE: 2024-10-11
Description: The code is modified to handle cases where images have no annotation labels.
"""

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

def compute_IoU_cIoU(input_json, gti_annotations):
    # Ground truth annotations
    coco_gt = COCO(gti_annotations)

    # load predicted annotations
    submission_file = json.loads(open(input_json).read())
    coco_dt = coco_gt.loadRes(submission_file)


    image_ids = coco_gt.getImgIds(catIds=coco_gt.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    pss = []
    for image_id in bar:
        # retrieve an image
        img = coco_gt.loadImgs(image_id)[0]

        # get GT mask and number of vertices
        annotation_ids = coco_gt.getAnnIds(imgIds=img['id'])
        N_GT = 0
        if len(annotation_ids) > 0:
            annotations = coco_gt.loadAnns(annotation_ids)
            for _idx, annotation in enumerate(annotations):
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)
                if _idx == 0:
                    mask_gt = m.reshape((img['height'], img['width']))
                    N_GT = len(annotation['segmentation'][0]) // 2
                else:
                    mask_gt = mask_gt + m.reshape((img['height'], img['width']))
                    N_GT = N_GT + len(annotation['segmentation'][0]) // 2
        else:
            mask_gt = np.zeros((img['height'], img['weight']), dtype=np.uint8)
        mask_gt = mask_gt != 0

        # get Predicted mask and number of vertices
        annotation_ids = coco_dt.getAnnIds(imgIds=img['id'])
        N = 0
        if len(annotation_ids) > 0:
            annotations = coco_dt.loadAnns(annotation_ids)
            for _idx, annotation in enumerate(annotations):
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)
                if _idx == 0:
                    mask = m.reshape((img['height'], img['width']))
                    N = len(annotation['segmentation'][0]) // 2
                else:
                    mask = mask + m.reshape((img['height'], img['width']))
                    N = N + len(annotation['segmentation'][0]) // 2
        else:
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
        mask = mask != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        pss.append(ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f, ps:%2.4f" % (np.mean(list_iou), np.mean(list_ciou), np.mean(pss)))
        bar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
