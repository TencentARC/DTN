import torch
import os
import random
import numpy as np
import shutil
from timm.models import resume_checkpoint


from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import logging
_logger = logging.getLogger(__name__)


class ModelEma_Norm(nn.Module):
    
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # make a copy of the model for accumulating moving average of weights
        super(ModelEma_Norm, self).__init__()
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                if k.endswith('iter') or k.endswith('norm_weight'):
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
def save_checkpoint(model_dir,epoch):
    #epoch = state['epoch']
    path = os.path.join(model_dir, 'model.pth-' + str(epoch))
    #torch.save(state, path)
    checkpoint_file = os.path.join(model_dir, 'checkpoint')
    checkpoint = open(checkpoint_file, 'w+')
    checkpoint.write('model_checkpoint_path:%s\n' % path)
    checkpoint.close()
    #if is_best:
    #    shutil.copyfile(path, os.path.join(model_dir, 'model-best.pth'))


def load_state(model_dir, model, optimizer=None, loss_scaler=None):
    if not os.path.exists(model_dir + '/last.pth.tar'):
        _logger.info("=> no checkpoint found at '{}', train from scratch".format(model_dir))
        return 0
    else:
        model_path = model_dir + '/last.pth.tar'
        resume_epoch = resume_checkpoint(
            model, model_path,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=True)
        return resume_epoch

def load_state_epoch(model_dir, model, optimizer=None, loss_scaler=None):
    
    model_path = model_dir
    resume_epoch = resume_checkpoint(
        model, model_path,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        log_info=True)
    return resume_epoch