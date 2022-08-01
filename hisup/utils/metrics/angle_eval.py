"""
This is the code from https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning.
@misc{girard2020polygonal,
    title={Polygonal Building Segmentation by Frame Field Learning},
    author={Nicolas Girard and Dmitriy Smirnov and Justin Solomon and Yuliya Tarabalka},
    year={2020},
    eprint={2004.14875},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
"""

import shapely
import shapely.geometry
import shapely.affinity
import shapely.ops
import shapely.prepared
import shapely.validation

import numpy as np
import random
import multiprocess
from multiprocess import Pool
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection



class ContourEval:
    def __init__(self, coco_gt, coco_dt):
        """
        @param coco_gt: coco object with ground truth annotations
        @param coco_dt: coco object with detection results
        """
        self.coco_gt = coco_gt  # ground truth COCO API
        self.coco_dt = coco_dt  # detections COCO API

        self.img_ids = sorted(coco_gt.getImgIds())
        self.cat_ids = sorted(coco_dt.getCatIds())

    def evaluate(self, pool=None):
        gts = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=self.img_ids))
        dts = self.coco_dt.loadAnns(self.coco_dt.getAnnIds(imgIds=self.img_ids))

        _gts = defaultdict(list)  # gt for evaluation
        _dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            _gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            _dts[dt['image_id'], dt['category_id']].append(dt)
        evalImgs = defaultdict(list)  # per-image per-category evaluation results

        # Compute metric
        args_list = []
        # i = 1000
        for img_id in self.img_ids:
            for cat_id in self.cat_ids:
                gts = _gts[img_id, cat_id]
                dts = _dts[img_id, cat_id]
                args_list.append((gts, dts))
                # i -= 1
            # if i <= 0:
            #     break

        if pool is None:
            measures_list = []
            for args in tqdm(args_list, desc="Contour metrics"):
                measures_list.append(compute_contour_metrics(args))
        else:
            measures_list = list(tqdm(pool.imap(compute_contour_metrics, args_list), desc="Contour metrics", total=len(args_list)))
        measures_list = [measure for measures in measures_list for measure in measures]  # Flatten list
        # half_tangent_cosine_similarities_list, edge_distances_list = zip(*measures_list)
        # half_tangent_cosine_similarities_list = [item for item in half_tangent_cosine_similarities_list if item is not None]
        measures_list = [value for value in measures_list if value is not None]
        max_angle_diffs = np.array(measures_list)
        max_angle_diffs = max_angle_diffs * 180 / np.pi  # Convert to degrees

        return max_angle_diffs

def compute_contour_metrics(gts_dts):
    gts, dts = gts_dts
    gt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in gts
                   for coords in ann["segmentation"]]
    dt_polygons = [shapely.geometry.Polygon(np.array(coords).reshape(-1, 2)) for ann in dts
                   for coords in ann["segmentation"]]
    fixed_gt_polygons = fix_polygons(gt_polygons, buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
    fixed_dt_polygons = fix_polygons(dt_polygons)
    # cosine_similarities, edge_distances = \
    #     polygon_utils.compute_polygon_contour_measures(dt_polygons, gt_polygons, sampling_spacing=2.0, min_precision=0.5,
    #                                                    max_stretch=2)
    max_angle_diffs = compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)

    return max_angle_diffs

def compute_polygon_contour_measures(pred_polygons: list, gt_polygons: list, sampling_spacing: float, min_precision: float, max_stretch: float, metric_name: str="cosine", progressbar=False):
    """
    pred_polygons are sampled with sampling_spacing before projecting those sampled points to gt_polygons.
    Then the
    @param pred_polygons:
    @param gt_polygons:
    @param sampling_spacing:
    @param min_precision: Polygons in pred_polygons must have a precision with gt_polygons above min_precision to be included in further computations
    @param max_stretch:  Exclude edges that have been stretched by the projection more than max_stretch from further computation
    @param metric_name: Metric type, can be "cosine" or ...
    @return:
    """
    assert isinstance(pred_polygons, list), "pred_polygons should be a list"
    assert isinstance(gt_polygons, list), "gt_polygons should be a list"
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return np.array([]), [], []
    assert isinstance(pred_polygons[0], shapely.geometry.Polygon), \
        f"Items of pred_polygons should be of type shapely.geometry.Polygon, not {type(pred_polygons[0])}"
    assert isinstance(gt_polygons[0], shapely.geometry.Polygon), \
        f"Items of gt_polygons should be of type shapely.geometry.Polygon, not {type(gt_polygons[0])}"
    gt_polygons = shapely.geometry.collection.GeometryCollection(gt_polygons)
    pred_polygons = shapely.geometry.collection.GeometryCollection(pred_polygons)
    # Filter pred_polygons to have at least a precision with gt_polygons of min_precision
    filtered_pred_polygons = [pred_polygon for pred_polygon in pred_polygons if min_precision < pred_polygon.intersection(gt_polygons).area / pred_polygon.area]
    # Extract contours of gt polygons
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
    # Measure metric for each pred polygon
    if progressbar:
        process_id = int(multiprocess.current_process().name[-1])
        iterator = tqdm(filtered_pred_polygons, desc="Contour measure", leave=False, position=process_id)
    else:
        iterator = filtered_pred_polygons
    half_tangent_max_angles = [compute_contour_measure(pred_polygon, gt_contours, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon in iterator]
    return half_tangent_max_angles

def fix_polygons(polygons, buffer=0.0):
    polygons = [
        geom if geom.is_valid else geom.buffer(0) for geom in polygons
    ]
    polygons_geom = shapely.ops.unary_union(polygons)  # Fix overlapping polygons
    polygons_geom = polygons_geom.buffer(buffer)  # Fix self-intersecting polygons and other things
    fixed_polygons = []
    if polygons_geom.geom_type == "MultiPolygon":
        for poly in polygons_geom:
            fixed_polygons.append(poly)
    elif polygons_geom.geom_type == "Polygon":
        fixed_polygons.append(polygons_geom)
    else:
        raise TypeError(f"Geom type {polygons_geom.geom_type} not recognized.")
    return fixed_polygons

def compute_contour_measure(pred_polygon, gt_contours, sampling_spacing, max_stretch, metric_name="cosine"):
    pred_contours = shapely.geometry.GeometryCollection([pred_polygon.exterior, *pred_polygon.interiors])
    sampled_pred_contours = sample_geometry(pred_contours, sampling_spacing)
    # Project sampled contour points to ground truth contours
    projected_pred_contours = project_onto_geometry(sampled_pred_contours, gt_contours)
    contour_measures = []
    for contour, proj_contour in zip(sampled_pred_contours, projected_pred_contours):
        coords = np.array(contour.coords[:])
        proj_coords = np.array(proj_contour.coords[:])
        edges = coords[1:] - coords[:-1]
        proj_edges = proj_coords[1:] - proj_coords[:-1]
        # Remove edges with a norm of zero
        edge_norms = np.linalg.norm(edges, axis=1)
        proj_edge_norms = np.linalg.norm(proj_edges, axis=1)
        norm_valid_mask = 0 < edge_norms * proj_edge_norms
        edges = edges[norm_valid_mask]
        proj_edges = proj_edges[norm_valid_mask]
        edge_norms = edge_norms[norm_valid_mask]
        proj_edge_norms = proj_edge_norms[norm_valid_mask]
        # Remove edge that have stretched more than max_stretch (invalid projection)
        stretch = edge_norms / proj_edge_norms
        stretch_valid_mask = np.logical_and(1 / max_stretch < stretch, stretch < max_stretch)
        edges = edges[stretch_valid_mask]
        if edges.shape[0] == 0:
            # Invalid projection for the whole contour, skip it
            continue
        proj_edges = proj_edges[stretch_valid_mask]
        edge_norms = edge_norms[stretch_valid_mask]
        proj_edge_norms = proj_edge_norms[stretch_valid_mask]
        scalar_products = np.abs(np.sum(np.multiply(edges, proj_edges), axis=1) / (edge_norms * proj_edge_norms))
        try:
            contour_measures.append(scalar_products.min())
        except ValueError:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [contour])
            plot_geometries(ax[1], [proj_contour])
            plot_geometries(ax[2], gt_contours)
            fig.tight_layout()
            plt.show()
    if len(contour_measures):
        min_scalar_product = min(contour_measures)
        measure = np.arccos(min_scalar_product)
        return measure
    else:
        return None

def sample_geometry(geom, density):
    """
    Sample edges of geom with a homogeneous density.
    @param geom:
    @param density:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        sampled_geom = shapely.geometry.GeometryCollection([sample_geometry(g, density) for g in geom])

        # toc = time.time()
        # print(f"sample_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        sampled_exterior = sample_geometry(geom.exterior, density)
        sampled_interiors = [sample_geometry(interior, density) for interior in geom.interiors]
        sampled_geom = shapely.geometry.Polygon(sampled_exterior, sampled_interiors)
    elif isinstance(geom, shapely.geometry.LineString):
        sampled_x = []
        sampled_y = []
        coords = np.array(geom.coords[:])
        lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
        for i in range(len(lengths)):
            start = geom.coords[i]
            end = geom.coords[i + 1]
            length = lengths[i]
            num = max(1, int(round(length / density))) + 1
            x_seq = np.linspace(start[0], end[0], num)
            y_seq = np.linspace(start[1], end[1], num)
            if 0 < i:
                x_seq = x_seq[1:]
                y_seq = y_seq[1:]
            sampled_x.append(x_seq)
            sampled_y.append(y_seq)
        sampled_x = np.concatenate(sampled_x)
        sampled_y = np.concatenate(sampled_y)
        sampled_coords = zip(sampled_x, sampled_y)
        sampled_geom = shapely.geometry.LineString(sampled_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return sampled_geom

def project_onto_geometry(geom, target, pool: Pool=None):
    """
    Projects all points from line_string onto target.
    @param geom:
    @param target:
    @param pool:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        if pool is None:
            projected_geom = [project_onto_geometry(g, target, pool=pool) for g in geom]
        else:
            partial_project_onto_geometry = partial(project_onto_geometry, target=target)
            projected_geom = pool.map(partial_project_onto_geometry, geom)
        projected_geom = shapely.geometry.GeometryCollection(projected_geom)

        # toc = time.time()
        # print(f"project_onto_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        projected_exterior = project_onto_geometry(geom.exterior, target)
        projected_interiors = [project_onto_geometry(interior, target) for interior in geom.interiors]
        try:
            projected_geom = shapely.geometry.Polygon(projected_exterior, projected_interiors)
        except shapely.errors.TopologicalError as e:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [geom])
            plot_geometries(ax[1], target)
            plot_geometries(ax[2], [projected_exterior, *projected_interiors])
            fig.tight_layout()
            plt.show()
            raise e
    elif isinstance(geom, shapely.geometry.LineString):
        projected_coords = [point_project_onto_geometry(coord, target) for coord in geom.coords]
        projected_geom = shapely.geometry.LineString(projected_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return projected_geom

def point_project_onto_geometry(coord, target):
    point = shapely.geometry.Point(coord)
    _, projected_point = shapely.ops.nearest_points(point, target)
    # dist = point.distance(projected_point)
    return projected_point.coords[0]

def plot_geometries(axis, geometries, linewidths=1, markersize=3):
    if len(geometries):
        patches = []
        for i, geometry in enumerate(geometries):
            if geometry.geom_type == "Polygon":
                polygon = shapely.geometry.Polygon(geometry)
                if not polygon.is_empty:
                    patch = PolygonPatch(polygon)
                    patches.append(patch)
                axis.plot(*polygon.exterior.xy, marker="o", markersize=markersize)
                for interior in polygon.interiors:
                    axis.plot(*interior.xy, marker="o", markersize=markersize)
            elif geometry.geom_type == "LineString" or geometry.geom_type == "LinearRing":
                axis.plot(*geometry.xy, marker="o", markersize=markersize)
            else:
                raise NotImplementedError(f"Geom type {geometry.geom_type} not recognized.")
        random.seed(1)
        colors = random.choices([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ], k=len(patches))
        edgecolors = np.array(colors)
        facecolors = edgecolors.copy()
        p = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
        axis.add_collection(p)