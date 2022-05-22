from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim



try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay_baselr(model, weight_decay=1e-5, base_lr=0.0005, skip_list=(), skip2=(),lower_lr=()):
    decay_lr = []
    no_decay_lr = []
    decay_lower_lr = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or name in skip2:
            no_decay_lr.append(param)
        elif name in lower_lr:
            decay_lower_lr.append(param)
        else:
            decay_lr.append(param)
    return [
        {'params': no_decay_lr,  'weight_decay': 0., 'lr': base_lr},
        {'params': decay_lower_lr,  'weight_decay': weight_decay, 'lr': 0.1*base_lr},
        {'params': decay_lr, 'weight_decay': weight_decay, 'lr': base_lr}
        ]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer_norm(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model: nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):
    """ Create an optimizer.
    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller
    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        learning_rate: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through
    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay2'):
            skip = model.no_weight_decay2()
        if hasattr(model, 'get_gating_param'):
            skip2 = model.get_gating_param()
        if hasattr(model, 'get_position_param'):
            lower_lr = model.get_position_param()
        parameters = add_weight_decay_baselr(model, weight_decay, learning_rate, skip, skip2,lower_lr)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    #opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    opt_args = dict(weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)

    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)

    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError



    return optimizer
