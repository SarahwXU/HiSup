import os
import torch
import argparse
import logging

from skimage import io
from parsing.config import cfg
from parsing.detector import BuildingDetector
from parsing.dataset.build import build_transform
from parsing.utils.comm import to_single_device
from parsing.utils.logger import setup_logger
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.utils.visualizer import show_polygons


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--img",
                        required=True,
                        dest='img',
                        nargs='+',
                        help="path to test image")
    
    args = parser.parse_args()
    
    return args


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE

    model = BuildingDetector(cfg)
    model = model.to(device)

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
        _ = checkpointer.load()        
        model = model.eval()

    impaths = args.img
    transform = build_transform(cfg)

    for impath in impaths:
        image = io.imread(impath)
        image_tensor = transform(image.astype(float))[None].to(device)
        meta = {
            'filename': impath,
            'height': image.shape[0],
            'width': image.shape[1],
        }

        with torch.no_grad():
            output, _ = model(image_tensor, [meta])
            output = to_single_device(output, 'cpu')

        polygons = output['polys_pred'][0]
        show_polygons(image, polygons)
    

if __name__ == "__main__":
    args = parse_args()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('testing', output_dir)
    logger.info(args)
    test(cfg, args)