"""
The code is adopted from https://github.com/spgriffin/polis
"""

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as maskUtils
from shapely import geometry
from shapely.geometry import Polygon


def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we traverse the collection of points only once, 
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y, top_right_x - bot_left_x, top_right_y - bot_left_y]

def compare_polys(poly_a, poly_b):
    """Compares two polygons via the "polis" distance metric.
    See "A Metric for Polygon Comparison and Building Extraction
    Evaluation" by J. Avbelj, et al.
    Input:
        poly_a: A Shapely polygon.
        poly_b: Another Shapely polygon.
    Returns:
        The "polis" distance between these two polygons.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    dist = polis(bndry_a.coords, bndry_b)
    dist += polis(bndry_b.coords, bndry_a)
    return dist


def polis(coords, bndry):
    """Computes one side of the "polis" metric.
    Input:
        coords: A Shapley coordinate sequence (presumably the vertices
                of a polygon).
        bndry: A Shapely linestring (presumably the boundary of
        another polygon).
    
    Returns:
        The "polis" metric for this pair.  You usually compute this in
        both directions to preserve symmetry.
    """
    sum = 0.0
    for pt in (geometry.Point(c) for c in coords[:-1]): # Skip the last point (same as first)
        sum += bndry.distance(pt)
    return sum/float(2*len(coords))


class PolisEval():

    def __init__(self, cocoGt=None, cocoDt=None):
        self.cocoGt   = cocoGt
        self.cocoDt   = cocoDt
        self.evalImgs = defaultdict(list)
        self.eval     = {}
        self._gts     = defaultdict(list)
        self._dts     = defaultdict(list)
        self.stats    = []
        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))

    def _prepare(self):
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluateImg(self, imgId):
        gts = self._gts[imgId]
        dts = self._dts[imgId]

        if len(gts) == 0 or len(dts) == 0:
            return 0

        gt_bboxs = [bounding_box(np.array(gt['segmentation'][0]).reshape(-1,2)) for gt in gts]
        dt_bboxs = [bounding_box(np.array(dt['segmentation'][0]).reshape(-1,2)) for dt in dts]
        gt_polygons = [np.array(gt['segmentation'][0]).reshape(-1,2) for gt in gts]
        dt_polygons = [np.array(dt['segmentation'][0]).reshape(-1,2) for dt in dts]

        # IoU match
        iscrowd = [0] * len(gt_bboxs)
        # ious = maskUtils.iou(gt_bboxs, dt_bboxs, iscrowd)
        ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

        # compute polis
        img_polis_avg = 0
        num_sample = 0
        for i, gt_poly in enumerate(gt_polygons):
            matched_idx = np.argmax(ious[:,i])
            iou = ious[matched_idx, i]
            if iou > 0.5: # iouThres:
                polis = compare_polys(Polygon(gt_poly), Polygon(dt_polygons[matched_idx]))
                img_polis_avg += polis
                num_sample += 1

        if num_sample == 0:
            return 0
        else:
            return img_polis_avg / num_sample


    def evaluate(self):
        self._prepare()
        polis_tot = 0

        num_valid_imgs = 0
        for imgId in tqdm(self.imgIds):
            img_polis_avg = self.evaluateImg(imgId)

            if img_polis_avg == 0:
                continue
            else:
                polis_tot += img_polis_avg
                num_valid_imgs += 1
        
        polis_avg = polis_tot / num_valid_imgs

        print('average polis: %f' % (polis_avg))

        return polis_avg


    



    