# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(cfgdir, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = optim.AdamW(parameters,eps=cfgdir.hyperparams.optimizer_eps, betas=(cfgdir.hyperparams.optimizer_beta1,cfgdir.hyperparams.optimizer_beta2),
                              lr=cfgdir.hyperparams.base_lr, weight_decay=cfgdir.hyperparams.weight_decay)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
