from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER
cfg = CN()

cfg.MODEL = MODELS
cfg.DATASETS = DATASETS
cfg.SOLVER = SOLVER

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8

cfg.OUTPUT_DIR = "outputs/default"