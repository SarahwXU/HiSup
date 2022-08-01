import torch

from hisup.csrc.lib.afm_op import afm
from torch.utils.data.dataloader import default_collate

class Encoder(object):
    def __init__(self, cfg):
        self.target_h = cfg.DATASETS.TARGET.HEIGHT
        self.target_w = cfg.DATASETS.TARGET.WIDTH

    def __call__(self, annotations):
        targets = []
        metas   = []
        for ann in annotations:
            t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def _process_per_image(self, ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        junc_tag = ann['juncs_tag']
        jmap = torch.zeros((height, width), device=device, dtype=torch.long)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)

        edges_positive = ann['edges_positive']
        if len(edges_positive) == 0:
            afmap = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
        else:
            lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
            shape_info = torch.IntTensor([[0, lines.size(0), height, width]])
            afmap, label = afm(lines, shape_info.cuda(), height, width)

        xint, yint = junctions[:,0].long(), junctions[:,1].long()
        off_x = junctions[:,0] - xint.float()-0.5
        off_y = junctions[:,1] - yint.float()-0.5
        jmap[yint, xint] = junc_tag
        joff[0, yint, xint] = off_x
        joff[1, yint, xint] = off_y
        meta = {
            'junc': junctions,
            'junc_index': ann['juncs_index'],
            'bbox': ann['bbox'],
        }

        mask = ann['mask'].float()
        target = {
            'jloc': jmap[None],
            'joff': joff,
            'mask': mask[None],
            'afmap': afmap[0]
        }
        return target, meta