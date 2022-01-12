""" 
    modified from timm.optim
"""
from torch import optim as optim
import torch.nn as nn

def add_weight_decay_and_lr(model: nn.Module, weight_decay=1e-5, text_lr=5e-4, skip_list=()):
    # set text_lr for text model
    visual_decay = []
    visual_nodecay = []
    text_decay = []
    text_nodecay = []

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list
        visual = ((name.startswith("visual")) or ('visual' in name))
        if visual and no_decay:
            print('visual key:', name)
            visual_nodecay.append(param)
        elif visual:
            print('visual key:', name)
            visual_decay.append(param)
        elif not visual and no_decay:
            print('text key:', name)
            text_nodecay.append(param)
        else:
            print('text key:', name)
            text_decay.append(param)

    return [
        {'params': visual_decay, 'weight_decay': weight_decay},
        {'params': visual_nodecay, 'weight_decay': 0.},
        {'params': text_decay, 'weight_decay': weight_decay, 'lr': text_lr},
        {'params': text_nodecay, 'weight_decay': 0., 'lr': text_lr}]


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        if hasattr(args, "text_lr") and args.text_lr > 0:
            print("lr: {:.2g} text_lr: {:.2g}".format(args.lr, args.text_lr))
            parameters = add_weight_decay_and_lr(model, weight_decay, text_lr=args.text_lr, skip_list=skip)
        else: 
            parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer
