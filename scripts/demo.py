import torch
import argparse
import numpy as np

from skimage import io
from hisup.config import cfg
from hisup.detector import get_pretrained_model
from hisup.dataset.build import build_transform
from hisup.utils.comm import to_single_device
from hisup.utils.visualizer import show_polygons


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--dataset", 
                        required=False,
                        type=str,
                        choices=['crowdai', 'inria'],
                        default='crowdai',
                        help="parameters' source")

    parser.add_argument("--img",
                        required=True,
                        type=str,
                        help="path to test image")

    args = parser.parse_args()

    return args

def inference_no_patching(cfg, model, image, device):
    transform = build_transform(cfg)
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        'height': image.shape[0],
        'width': image.shape[1],
    }

    with torch.no_grad():
        output, _ = model(image_tensor, [meta])
        output = to_single_device(output, 'cpu')

    if len(output['polys_pred']) > 0:
        polygons = output['polys_pred'][0]
        show_polygons(image, polygons)
    else:
        print('No building polygons.')

def inference_with_patching(cfg, model, image, device):
    import scipy.ndimage
    from tqdm import tqdm
    from shapely.geometry import Polygon
    from skimage.measure import label, regionprops
    from hisup.utils.polygon import generate_polygon, juncs_in_bbox
    from hisup.utils.visualizer import viz_inria

    transform = build_transform(cfg)
    
    h_stride, w_stride = 400, 400
    h_crop, w_crop = cfg.DATASETS.ORIGIN.HEIGHT, cfg.DATASETS.ORIGIN.WIDTH
    h_img, w_img, _ = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
    count_mat = np.zeros([h_img, w_img])
    juncs_whole_img = []
    
    patch_weight = np.ones((h_crop + 2, w_crop + 2))
    patch_weight[0,:] = 0
    patch_weight[-1,:] = 0
    patch_weight[:,0] = 0
    patch_weight[:,-1] = 0
    
    patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
    patch_weight = patch_weight[1:-1,1:-1]

    for h_idx in tqdm(range(h_grids), desc='processing on image'):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            
            crop_img = image[y1:y2, x1:x2, :]
            crop_img_tensor = transform(crop_img.astype(float))[None].to(device)
            
            meta = {
                'height': crop_img.shape[0],
                'width': crop_img.shape[1],
                'pos': [x1, y1, x2, y2]
            }

            with torch.no_grad():
                output, _ = model(crop_img_tensor, [meta])
                output = to_single_device(output, 'cpu')

            juncs_pred = output['juncs_pred'][0]
            mask_pred = output['mask_pred'][0]
            juncs_pred += [x1, y1]
            juncs_whole_img.extend(juncs_pred.tolist())
            mask_pred *= patch_weight
            pred_whole_img += np.pad(mask_pred,
                                ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                (int(x1), int(pred_whole_img.shape[1] - x2))))
            count_mat[y1:y2, x1:x2] += patch_weight

    juncs_whole_img = np.array(juncs_whole_img)
    pred_whole_img = pred_whole_img / count_mat

    # match junction and seg results
    polygons = []
    props = regionprops(label(pred_whole_img > 0.5))
    for prop in tqdm(props, leave=False, desc='polygon generation'):
        y1, x1, y2, x2 = prop.bbox
        bbox = [x1, y1, x2, y2]
        select_juncs = juncs_in_bbox(bbox, juncs_whole_img, expand=8)
        poly, juncs_sa, _, _, juncs_index = generate_polygon(prop, pred_whole_img, \
                                                                    select_juncs, pid=0, test_inria=True)
        if juncs_sa.shape[0] == 0:
            continue
        
        if len(juncs_index) == 1:
            polygons.append(Polygon(poly))
        else:
            poly_ = Polygon(poly[juncs_index[0]], \
                            [poly[idx] for idx in juncs_index[1:]])
            polygons.append(poly_)

    viz_inria(image, polygons, cfg.OUTPUT_DIR, '')


def test(cfg, args):
    device = cfg.MODEL.DEVICE
    if not torch.cuda.is_available():
        device = 'cpu'

    image = io.imread(args.img)[:, :, :3]
    H, W = image.shape[:2]
    img_mean, img_std = [], []
    for i in range(image.shape[-1]):
        pixels = image[:, :, i].ravel()
        img_mean.append(np.mean(pixels))
        img_std.append(np.std(pixels))
    cfg.DATASETS.IMAGE.PIXEL_MEAN = img_mean
    cfg.DATASETS.IMAGE.PIXEL_STD  = img_std

    patching = False
    if H > 512 or W > 512:
        patching = True
        cfg.DATASETS.ORIGIN.HEIGHT = 512 if H > 512 else H
        cfg.DATASETS.ORIGIN.WIDTH = 512 if W > 512 else W
    else:
        cfg.DATASETS.ORIGIN.HEIGHT = H
        cfg.DATASETS.ORIGIN.WIDTH = W

    model = get_pretrained_model(cfg, args.dataset, device, pretrained=True)
    model = model.to(device)

    if not patching:
        inference_no_patching(cfg, model, image, device)
    else:
        inference_with_patching(cfg, model, image, device)


if __name__ == "__main__":
    args = parse_args()
    test(cfg, args)
