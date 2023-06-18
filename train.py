import math
from typing import Dict, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BaseConfig, BaselineConfig
from src.data_utils.cirr_dataset import CIRRDataset, targetpad_transform, squarepad_transform, ImageIndexingDataset
from src.model.baseline import BaselineModel
from src.model.modules.opts import element_wise_sum
from src.third_party import clip
from src.utils.constants import LOGGER

"""
Training pipeline:

DataLoader -> Model -> Optimizer & Scheduler -> Training 
"""


def register_dataloader(
  config: Union[BaseConfig, BaselineConfig],
  preprocess: Callable
) -> Dict[str, DataLoader]:
  LOGGER.info('Registering data loaders ...')
  val_index_dataset = ImageIndexingDataset('val', preprocess)
  val_index_dataloader = DataLoader(val_index_dataset, batch_size=config.index_batch_size,
                                    pin_memory=True, num_workers=4, shuffle=False)

  val_dataset = CIRRDataset('val', preprocess)
  val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size_per_gpu,
                              pin_memory=True, num_workers=4, shuffle=False)

  train_dataset = CIRRDataset('train', preprocess)
  train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size_per_gpu,
                                pin_memory=True, num_workers=4)

  return {
    'train': train_dataloader,
    'val': val_dataloader,
    'val_index': val_index_dataloader,
  }


def register_model(config: Union[BaseConfig, BaselineConfig]):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  if config.use_clip:
    clip_model, _ = clip.load(config.clip_model_name, device=device, jit=False)

    if config.torch_dtype == 'fp16':
      clip_model.float()

  if config.model_type == 'baseline':
    model = BaselineModel(config, clip_model)

  return {
    'model': model,
    'device': device
  }


def register_optimizer(
  config: Union[BaseConfig, BaselineConfig],
  model: nn.Module,
  train_dataloader: DataLoader
):
  LOGGER.info('Registering optimizer ...')
  optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(config.adam_betas[0], config.adam_betas[1]),
                    eps=config.adam_epsilon)

  t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
  warmup_steps = math.ceil(t_total * config.warmup_proportion)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

  return {
    'optimizer': optimizer,
    'scheduler': scheduler
  }


def training_one_step(
  config: Union[BaseConfig, BaselineConfig],
  batch: Dict[str, torch.Tensor],
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  scheduler,
  scaler: GradScaler = None,
) -> Dict[str, torch.Tensor]:
  if config.torch_dtype == 'fp16':
    assert scaler is not None

    with torch.cuda.amp.autocast():
      model_output = model(input_ids=batch['caption'].squeeze(dim=1), reference_feats=batch['reference_image'],
                           target_feats=batch['target_image'])
  else:
    model_output = model(input_ids=batch['caption'].squeeze(dim=1), reference_feats=batch['reference_image'],
                         target_feats=batch['target_image'])

  assert 'loss' in model_output, 'Model output should include `loss`.'

  loss = model_output['loss']

  if config.torch_dtype == 'fp16':
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scale = scaler.get_scale()
    scaler.update()
    optimizer.zero_grad()
    skip_lr_sched = (scale != scaler.get_scale())

    if not skip_lr_sched:
      scheduler.step()
  else:
    loss.backward()

    if config.max_grad_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

  return {'loss': loss}


def generate_index_features(dataloader: DataLoader, model: nn.Module, device):
  index_features, index_names = [], []

  for batch in tqdm(dataloader, desc=f'Indexing images ...', total=len(dataloader)):
    with torch.no_grad():
      index_feature = model.generate_embedding(batch['image'].to(device, non_blocking=True), embed_type='image')
    index_name = batch['image_name']

    index_features.append(index_feature.detach().cpu())
    index_names.extend(index_name)

  index_features = torch.cat(index_features, dim=0)

  return index_features, index_names


def evaluate(dataloader, model, index_features, index_names, device):
  all_predicted_features = []
  all_target_names, all_group_members, all_reference_names = [], [], []

  for batch in tqdm(dataloader, desc='Generating retrieval features ...', total=len(dataloader)):
    batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    with torch.no_grad():
      text_features, image_features = model.generate_embedding(batch['caption'].squeeze(dim=1), embed_type='text'), \
                                      model.generate_embedding(batch['reference_image'], embed_type='image')

    predicted_features = element_wise_sum(image_features, text_features)

    all_predicted_features.append(predicted_features.detach().cpu())
    all_target_names.extend(batch['target_hard_name'])
    all_group_members.extend(np.array(batch['group_member']).T.tolist())
    all_reference_names.extend(batch['reference_name'])

  # [data_size x dim]
  all_predicted_features = torch.cat(all_predicted_features, dim=0)
  all_predicted_features, index_features = all_predicted_features.to(device, non_blocking=True), index_features.to(device, non_blocking=True)

  # *** Evaluation phrase. ***
  index_features = F.normalize(index_features, dim=-1)  # .float()

  distances = all_predicted_features @ index_features.T
  sorted_indices = torch.argsort(distances, dim=-1, descending=True).cpu()
  sorted_index_names = np.array(index_names)[sorted_indices]

  reference_mask = torch.tensor(
    sorted_index_names != np.repeat(np.array(all_reference_names), len(index_names)).reshape(len(all_target_names),
                                                                                             -1))
  sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                  sorted_index_names.shape[1] - 1)
  # Compute the ground-truth labels wrt the predictions
  labels = torch.tensor(
    sorted_index_names == np.repeat(np.array(all_target_names), len(index_names) - 1).reshape(len(all_target_names),
                                                                                              -1))

  # Compute the subset predictions and ground-truth labels
  group_members = np.array(all_group_members)
  group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
  group_labels = labels[group_mask].reshape(labels.shape[0], -1)

  assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(all_target_names)).int())
  assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(all_target_names)).int())

  # Compute the metrics
  recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
  recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
  recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
  recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
  group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
  group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
  group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

  LOGGER.info(f'Recall@1: {recall_at1}')
  LOGGER.info(f'Recall@5: {recall_at5}')
  LOGGER.info(f'Recall@10: {recall_at10}')
  LOGGER.info(f'Recall@50: {recall_at50}')
  LOGGER.info(f'Group Recall@1: {group_recall_at1}')
  LOGGER.info(f'Group Recall@2: {group_recall_at2}')
  LOGGER.info(f'Group Recall@3: {group_recall_at3}')

  # return {
  #   'recall@1': recall_at1,
  #   'recall@5': recall_at1,
  #   'recall@10': recall_at1,
  #   'recall@50': recall_at1,
  #   'group_recall@1': recall_at1,
  #   'recall@1': recall_at1,
  # }
  #


def main(config: Union[BaseConfig, BaselineConfig]):
  torch.cuda.empty_cache()
  preprocess = targetpad_transform(1.25, config.clip_input_dim) if config.preprocess_func == 'targetpad' \
    else squarepad_transform(config.clip_input_dim)

  all_dataloaders = register_dataloader(config, preprocess)
  model, device = register_model(config).values()

  optimizer, scheduler = register_optimizer(config, model, all_dataloaders['train']).values()

  if config.torch_dtype == 'fp16':
    scaler = GradScaler()

  model.zero_grad()
  model.to(device)

  global_steps = 0
  best_recall_at_5 = -999

  for epoch in range(1, config.num_train_epochs + 1):
    torch.cuda.empty_cache()
    iter_bar = tqdm(all_dataloaders['train'], desc='Training phrase')
    iter_loss = 0.
    model.train()

    for step, batch in enumerate(iter_bar):
      batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
      model_output = training_one_step(config, batch, model, optimizer, scheduler,
                                       scaler) if config.torch_dtype == 'fp16' \
        else training_one_step(config, batch, model, optimizer, scheduler)
      loss = model_output['loss']
      iter_loss += loss.item()
      LOGGER.info(loss.item())

      global_steps += 1
      iter_bar.set_postfix({
        'epoch': f'{epoch}',
        'global_steps': f'{global_steps}',
        "learning_rate": f'{scheduler.get_last_lr()[0]:.10f}',
        "mean_loss": f"{iter_loss / (step + 1) * config.gradient_accumulation_steps:.5f}",
        "last_loss": f"{loss.item() * config.gradient_accumulation_steps:.5f}",
      })

    model.eval()
    index_features, index_names = generate_index_features(all_dataloaders['val_index'], model, device)
    # index_features, index_names = generate_index_features(val_index_dataloader, model, device)
    evaluate(all_dataloaders['val'], model, index_features, index_names, device)
    # evaluate(val_dataloader, model, index_features, index_names, device)


if __name__ == '__main__':
  config = BaselineConfig()
  main(config)

  pass
