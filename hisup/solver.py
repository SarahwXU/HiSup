import torch
import math

def make_optimizer(cfg, model):

    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr=cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    cfg.SOLVER.BASE_LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'ADAM' or 'ADAMcos':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    elif cfg.SOLVER.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    else:
        raise NotImplementedError()
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.OPTIMIZER == 'ADAMcos':
        t = cfg.SOLVER.STATIC_STEP
        max_ep = cfg.SOLVER.MAX_EPOCH
        lambda1 = lambda epoch: 1 if epoch < t else 0.00001 \
            if 0.5*(1+math.cos(math.pi*(epoch-t)/(max_ep-t))) < 0.00001 \
            else 0.5*(1+math.cos(math.pi*(epoch-t)/(max_ep-t)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)