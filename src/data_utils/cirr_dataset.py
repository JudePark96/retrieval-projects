import json
import logging
import os.path
import sys
from typing import Union, Dict, List

import PIL
import loguru as loguru
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import BaseConfig, BaselineConfig
from src.third_party import clip
from src.utils.constants import LOGGER


class Base(Dataset):
  def __init__(
    self,
    config: Union[BaseConfig, BaselineConfig],
    split: str,
    preprocess: callable,
  ) -> None:
    super().__init__()
    self.config = config
    self.preprocess = preprocess
    self.split = split

    if split not in ['test1', 'train', 'val']:
      raise ValueError("split should be in ['test1', 'train', 'val']")

    # get triplets made by (reference_image, target_image, relative caption)
    caption_file = f'cap.rc2.{split}.json'

    with open(os.path.join(config.base_path, f'cirr/captions/{caption_file}')) as f:
      self.triplets = json.load(f)

    # get a mapping from image name to relative path
    with open(os.path.join(config.base_path, f'cirr/image_splits/split.rc2.{split}.json')) as f:
      self.name_to_relpath = json.load(f)

    LOGGER.info(f"CIRR {split} dataset initialized")


class ImageIndexingDataset(Base):
  def __init__(self, config: Union[BaseConfig, BaselineConfig], split: str, preprocess: callable) -> None:
    super().__init__(config, split, preprocess)

  def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
    image_name = list(self.name_to_relpath.keys())[idx]
    image_path = os.path.join(self.config.base_path, self.name_to_relpath[image_name])
    im = Image.open(image_path)
    image = self.preprocess(im)

    return {
      'image_name': image_name,
      'image': image,
    }

  def __len__(self) -> int:
    return len(self.name_to_relpath)


class CIRRDataset(Base):
  def __init__(self, config: Union[BaseConfig, BaselineConfig], split: str, preprocess: callable) -> None:
    super().__init__(config, split, preprocess)

  def __getitem__(
    self,
    idx: int
  ) -> Dict[str, Union[
    str,
    List[int],
    np.ndarray,
    torch.Tensor,
  ]]:
    group_members = self.triplets[idx]['img_set']['members']
    reference_name = self.triplets[idx]['reference']
    target_hard_name = self.triplets[idx]['target_hard']
    rel_caption = self.triplets[idx]['caption']

    reference_image_path = os.path.join(self.config.base_path, f'{self.name_to_relpath[reference_name]}')
    reference_image = self.preprocess(Image.open(reference_image_path))
    target_image_path = os.path.join(self.config.base_path, f'{self.name_to_relpath[target_hard_name]}')
    target_image = self.preprocess(Image.open(target_image_path))

    if self.split == 'test1':
      # TODO => 테스트 데이터 환경에서도 동작하도록 수정할 것.
      pair_id = self.triplets[idx]['pairid']
      return pair_id, reference_name, rel_caption, group_members

    return {
      'reference_name': reference_name,
      'target_hard_name': target_hard_name,
      'group_member': group_members,
      'caption': clip.tokenize(rel_caption, context_length=77, truncate=True).squeeze(dim=0),
      'reference_image': reference_image,
      'target_image': target_image,
    }

  def __len__(self) -> int:
    return len(self.triplets)
