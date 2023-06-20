import math
from typing import Dict, Union, Callable, Tuple

import torch
import wandb
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import BaselineConfig, BaseConfig
from src.data_utils.cirr_dataset import CIRRDataset, targetpad_transform, squarepad_transform, ImageIndexingDataset
from src.evaluation import generate_index_features, evaluate
from src.model.baseline import BaselineModel
from src.third_party import clip
from src.utils.constants import LOGGER

"""
Training pipeline:

DataLoader -> Model -> Optimizer & Scheduler -> Training 
"""


def register_dataloader(
  config: Union[BaseConfig, BaselineConfig],
  preprocess: Callable
) -> Tuple[DataLoader]:
  LOGGER.info('Registering data loaders ...')
  val_index_dataset = ImageIndexingDataset('val', preprocess)
  val_index_dataloader = DataLoader(val_index_dataset, batch_size=config.index_batch_size,
                                    pin_memory=True, num_workers=8, shuffle=False)

  val_dataset = CIRRDataset('val', preprocess)
  val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size_per_gpu,
                              pin_memory=True, num_workers=8, shuffle=False)

  train_dataset = CIRRDataset('train', preprocess)
  train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size_per_gpu,
                                pin_memory=True, num_workers=8)
  return train_dataloader, val_dataloader, val_index_dataloader


def register_model(config: Union[BaseConfig, BaselineConfig]):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  if config.use_clip:
    clip_model, _ = clip.load(config.clip_model_name, device=device, jit=False)

    if config.torch_dtype == 'fp16':
      clip_model.float()

  if config.model_type == 'baseline':
    model = BaselineModel(config, clip_model)

  return model, device


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

  return optimizer, scheduler


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


def main(config: Union[BaseConfig, BaselineConfig]):
  preprocess = targetpad_transform(1.25, config.clip_input_dim) if config.preprocess_func == 'targetpad' \
    else squarepad_transform(config.clip_input_dim)

  train_dataloader, val_dataloader, val_index_dataloader = register_dataloader(config, preprocess)
  model, device = register_model(config)

  optimizer, scheduler = register_optimizer(config, model, train_dataloader)

  if config.torch_dtype == 'fp16':
    scaler = GradScaler()

  model.zero_grad()
  model.to(device)

  global_steps = 0
  best_recall_at_5 = -999

  wandb.init(
    # set the wandb project where this run will be logged
    project=config.project_name,
    name=f'{config.model_type}-bs{config.train_batch_size_per_gpu}-seed{config.seed}-clip{config.clip_model_name}',
    # track hyperparameters and run metadata
    config=config.__dict__,
  )

  for epoch in range(1, config.num_train_epochs + 1):
    iter_bar = tqdm(train_dataloader, desc='Training phrase')
    iter_loss = 0.
    model.train()
    for step, batch in enumerate(iter_bar):
      batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
      model_output = training_one_step(config, batch, model, optimizer, scheduler,
                                       scaler) if config.torch_dtype == 'fp16' \
        else training_one_step(config, batch, model, optimizer, scheduler)

      loss = model_output['loss']
      iter_loss += loss.item()

      if (step + 1) % config.logging_per_step == 0:
        wandb.log({"loss": loss.item()})

      global_steps += 1
      iter_bar.set_postfix({
        'epoch': f'{epoch}',
        'global_steps': f'{global_steps}',
        "learning_rate": f'{scheduler.get_last_lr()[0]:.10f}',
        "mean_loss": f"{iter_loss / (step + 1) * config.gradient_accumulation_steps:.5f}",
        "last_loss": f"{loss.item() * config.gradient_accumulation_steps:.5f}",
      })

    model.eval()
    index_features, index_names = generate_index_features(val_index_dataloader, model, device)
    eval_output_dicts = evaluate(val_dataloader, model, index_features, index_names, device)
    wandb.log(eval_output_dicts)


if __name__ == '__main__':
  config = BaselineConfig()
  main(config)
