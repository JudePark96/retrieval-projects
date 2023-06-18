from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BaseConfig, BaselineConfig
from test2 import ImageIndexingDataset, CIRRDataset
from src.model.modules.opts import element_wise_sum
from src.third_party import clip
from src.utils.constants import LOGGER
from src.utils.transforms import squarepad_transform


@torch.no_grad()
def zero_shot_eval_main(config: Union[BaseConfig, BaselineConfig]) -> None:
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  clip_model, preprocess = clip.load(config.clip_model_name, device=device, jit=False)
  clip_model = clip_model.to(device)
  clip_model.eval()

  input_dim = clip_model.visual.input_resolution
  LOGGER.info(input_dim)
  # preprocess = targetpad_transform(1.25, input_dim)
  preprocess = squarepad_transform(input_dim)

  index_dataset = ImageIndexingDataset(config, 'val', preprocess)
  index_dataloader = DataLoader(index_dataset, batch_size=256, pin_memory=True, num_workers=48, shuffle=False)

  index_features, index_names = [], []

  for batch in tqdm(index_dataloader, desc=f'Indexing images ...', total=len(index_dataloader)):
    index_feature = clip_model.encode_image(batch['image'].to(device))
    index_name = batch['image_name']

    index_features.append(index_feature.detach().cpu())
    index_names.extend(index_name)

  index_features = torch.cat(index_features, dim=0)
  LOGGER.info(index_features.shape)

  # Note that indexing function should be CHANGED depending on model design.
  # index_features = dict(zip(index_names, index_features))

  eval_dataset = CIRRDataset(config, 'val', preprocess)
  eval_dataloader = DataLoader(eval_dataset, batch_size=128, pin_memory=True, num_workers=48)

  all_predicted_features = []
  all_target_names, all_group_members, all_reference_names = [], [], []

  for batch in tqdm(eval_dataloader, desc='Generating retrieval features ...', total=len(eval_dataloader)):
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    text_features, image_features = clip_model.encode_text(batch['caption']), \
                                    clip_model.encode_image(batch['reference_image'])

    predicted_features = element_wise_sum(image_features, text_features)

    all_predicted_features.append(predicted_features.detach().cpu())
    all_target_names.extend(batch['target_hard_name'])
    all_group_members.extend(np.array(batch['group_member']).T.tolist())
    all_reference_names.extend(batch['reference_name'])

  # [data_size x dim]
  all_predicted_features = torch.cat(all_predicted_features, dim=0)
  all_predicted_features, index_features = all_predicted_features.to(device), index_features.to(device)

  # *** Evaluation phrase. ***
  index_features = F.normalize(index_features, dim=-1)#.float()

  distances = all_predicted_features @ index_features.T
  sorted_indices = torch.argsort(distances, dim=-1, descending=True).cpu()
  sorted_index_names = np.array(index_names)[sorted_indices]

  reference_mask = torch.tensor(
    sorted_index_names != np.repeat(np.array(all_reference_names), len(index_names)).reshape(len(all_target_names), -1))
  sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                  sorted_index_names.shape[1] - 1)
  # Compute the ground-truth labels wrt the predictions
  labels = torch.tensor(
    sorted_index_names == np.repeat(np.array(all_target_names), len(index_names) - 1).reshape(len(all_target_names), -1))

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

  """
  TargetPad
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:106 | evaluation | zero_shot_eval_main |> Recall@1: 22.052140533924103 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:107 | evaluation | zero_shot_eval_main |> Recall@5: 51.925379037857056 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:108 | evaluation | zero_shot_eval_main |> Recall@10: 66.01291298866272 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:109 | evaluation | zero_shot_eval_main |> Recall@50: 88.18464279174805 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:110 | evaluation | zero_shot_eval_main |> Group Recall@1: 52.95383930206299 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:111 | evaluation | zero_shot_eval_main |> Group Recall@2: 74.21669363975525 
  2023-06-18T00:10:46.928+09:00 |     INFO | evaluation.py:112 | evaluation | zero_shot_eval_main |> Group Recall@3: 86.98875904083252 
  """

  """
  SquarePad
  2023-06-18T02:02:42.547+09:00 |     INFO | evaluation.py:102 | evaluation | zero_shot_eval_main |> Recall@1: 22.4348247051239 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:103 | evaluation | zero_shot_eval_main |> Recall@5: 51.71011686325073 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:104 | evaluation | zero_shot_eval_main |> Recall@10: 65.0562047958374 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:105 | evaluation | zero_shot_eval_main |> Recall@50: 87.46711015701294 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:106 | evaluation | zero_shot_eval_main |> Group Recall@1: 52.6189923286438 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:107 | evaluation | zero_shot_eval_main |> Group Recall@2: 74.168860912323 
  2023-06-18T02:02:42.548+09:00 |     INFO | evaluation.py:108 | evaluation | zero_shot_eval_main |> Group Recall@3: 86.22339367866516 
  """

  """
  CLIP
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:101 | evaluation | zero_shot_eval_main |> Recall@1: 21.669456362724304 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:102 | evaluation | zero_shot_eval_main |> Recall@5: 50.87299942970276 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:103 | evaluation | zero_shot_eval_main |> Recall@10: 65.39105772972107 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:104 | evaluation | zero_shot_eval_main |> Recall@50: 87.61062026023865 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:105 | evaluation | zero_shot_eval_main |> Group Recall@1: 51.327431201934814 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:106 | evaluation | zero_shot_eval_main |> Group Recall@2: 72.44678139686584 
  2023-06-18T01:53:24.310+09:00 |     INFO | evaluation.py:107 | evaluation | zero_shot_eval_main |> Group Recall@3: 85.98421216011047 
  """

if __name__ == '__main__':
  config = BaselineConfig()
  zero_shot_eval_main(config)
