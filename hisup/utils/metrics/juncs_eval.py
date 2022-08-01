"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{conv-mpn,
Title = {Conv-MPN: Convolutional Message Passing Neural Network for Structured Outdoor Architecture Reconstruction},
Author = {Fuyang, Zhang and Nelson, Nauata and Yasutaka, Furukawa},
Year = {2019},
journal = {arXiv preprint arXiv:1912.01756}
}
"""

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm
from rdp import rdp


def evaluate_junctions(input_json, gti_annotations):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)


    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    thresh = 5
    tps, fps = 0, 0
    num_gts = 0
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        dts = np.zeros((0,2))
        for ann in annotations:
            juncs = np.array(ann['segmentation'][0]).reshape(-1, 2)
            # juncs = rdp(juncs, 0.125)
            dts = np.append(dts, juncs, axis=0)

        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        gts = np.zeros((0,2))
        for ann in annotations:
            juncs = np.array(ann['segmentation'][0]).reshape(-1, 2)
            gts = np.append(gts, juncs[:-1], axis=0)
        
        found = [False] * len(gts)
        per_sample_tp = 0
        per_sample_fp = 0
        for i, det in enumerate(dts):

            # get closest gt
            near_gt = [0, 9999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt-det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt] 

            # hit (<= thresh) and not found yet 
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_tp += 1.0
                found[near_gt[0]] = True

            # not hit or already found
            else:
                per_sample_fp += 1.0

        recall = per_sample_tp / gts.shape[0]
        precision = per_sample_tp / (per_sample_tp + per_sample_fp + 1e-8)

        tps += per_sample_tp
        fps += per_sample_fp
        num_gts += gts.shape[0]

        bar.set_description("R: %2.4f, P: %2.4f" % (recall, precision))
        bar.refresh()

    R = tps / num_gts
    P = tps / (tps + fps + 1e-8)
    F1 = 2.0*P*R/(P+R+1e-8)

    print('Done!')
    print('Precision: %2.4f, Recall: %2.4f, F1-score: %2.4f' % (P, R, F1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    evaluate_junctions(input_json=dt_file,
                       gti_annotations=gt_file)