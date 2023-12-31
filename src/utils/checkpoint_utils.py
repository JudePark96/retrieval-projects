import os
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim

from typing import Any
from torch.nn.parallel import DistributedDataParallel


def save_checkpoints(
        save_checkpoints_dir, cur_epoch, cur_step, model, optimizer, scheduler, checkpoint_prefix=None
):
    # type: (str, int, int, nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR, str) -> None
    """
    Save checkpoints. Execute at the end of a iteration.

    Arguments:
        save_checkpoints_dir (str):
        cur_epoch (int):
        cur_step (int):optim.lr_scheduler.LambdaLR, str
        model (nn.Module):
        optimizer (optim.Optimizer):
        scheduler (optim.lr_scheduler.LambdaLR):
        checkpoint_prefix (str, optional):
    """

    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': cur_epoch,
        'step': cur_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    checkpoint_file = (f"{checkpoint_prefix}_" if checkpoint_prefix else "") + f"epoch_{cur_epoch}.ckpt"
    save_path = os.path.join(save_checkpoints_dir, checkpoint_file)

    # _use_new_zipfile_serialization=False is for torch version == 1.7
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)


def load_checkpoint_attribute_from_key(load_checkpoints_path, any_obj, key):
    # type: (str, Any, str) -> Any

    checkpoint = torch.load(load_checkpoints_path, map_location='cpu')

    value = checkpoint[key]
    if isinstance(value, int):
        any_obj = value
    else:
        any_obj.load_state_dict(value)

    return any_obj


def save_model_state_dict(save_state_dict_dir, state_dict_file_name, model, **kwargs):
    # type: (str, str, (nn.Module, DistributedDataParallel)) -> None
    """
    Save model state dict. Note this function is not for reproduce or continuing train.

    Arguments:
        model (nn.Module):
        save_state_dict_dir (str):
        state_dict_file_name (str):
    """

    model_to_save = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(save_state_dict_dir, state_dict_file_name)

    # _use_new_zipfile_serialization=False is for torch version == 1.7
    torch.save(
        {
            'model_state_dict': model_to_save.state_dict(),
            **kwargs
        # }, save_path, _use_new_zipfile_serialization=False)
        }, save_path)