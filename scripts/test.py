import os
import argparse
import logging

from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.utils.logger import setup_logger
from hisup.utils.checkpoint import DetectronCheckpointer
from tools.test_pipelines import TestPipeline

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--eval-type",
                        type=str,
                        help="evalutation type for the test results",
                        default="coco_iou",
                        choices=["coco_iou",  "boundary_iou", "polis"]
                        )

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    
    return args


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE
    model = BuildingDetector(cfg, test=True)
    model = model.to(device)

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
        _ = checkpointer.load()        
        model = model.eval()

    test_pipeline = TestPipeline(cfg, args.eval_type)
    test_pipeline.test(model)
    # test_pipeline.eval()


if __name__ == "__main__":
    args = parse_args()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('testing', output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    test(cfg, args)

