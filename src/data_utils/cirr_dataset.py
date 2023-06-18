import json
import os.path
from typing import Union, Dict

import PIL
import PIL.Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from src.third_party import clip
from src.utils.constants import LOGGER

base_path = '/opt/home/jude/workspace/llm-retrieval/'


def _convert_image_to_rgb(image):
  return image.convert("RGB")


class SquarePad:
  def __init__(self, size: int):
    self.size = size

  def __call__(self, image):
    w, h = image.size
    max_wh = max(w, h)
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = [hp, vp, hp, vp]
    return F.pad(image, padding, 0, 'constant')


class TargetPad:
  def __init__(self, target_ratio: float, size: int):
    self.size = size
    self.target_ratio = target_ratio

  def __call__(self, image):
    w, h = image.size
    actual_ratio = max(w, h) / min(w, h)
    if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
      return image
    scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
    hp = max(int((scaled_max_wh - w) / 2), 0)
    vp = max(int((scaled_max_wh - h) / 2), 0)
    padding = [hp, vp, hp, vp]
    return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
  return Compose([
    SquarePad(dim),
    Resize(dim, interpolation=PIL.Image.BICUBIC),
    CenterCrop(dim),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])


def targetpad_transform(target_ratio: float, dim: int):
  return Compose([
    TargetPad(target_ratio, dim),
    Resize(dim, interpolation=PIL.Image.BICUBIC),
    CenterCrop(dim),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
  ])


class Base(Dataset):
  def __init__(self, split: str, preprocess: callable):
    self.preprocess = preprocess
    self.split = split

    if split not in ['test1', 'train', 'val']:
      raise ValueError("split should be in ['test1', 'train', 'val']")

    caption_file = f'cap.rc2.{split}.json'

    with open(os.path.join(base_path, 'cirr_dataset', 'cirr', 'captions', caption_file)) as f:
      self.triplets = json.load(f)

    with open(os.path.join(base_path, 'cirr_dataset', 'cirr', 'image_splits', f'split.rc2.{split}.json')) as f:
      self.name_to_relpath = json.load(f)

    LOGGER.info(f"CIRR {split} dataset initialized")


class ImageIndexingDataset(Base):

  def __init__(self, split: str, preprocess: callable):
    super().__init__(split, preprocess)

  def __getitem__(self, idx: int):
    image_name = list(self.name_to_relpath.keys())[idx]
    image_path = os.path.join(base_path, 'cirr_dataset', self.name_to_relpath[image_name])
    im = PIL.Image.open(image_path)
    image = self.preprocess(im)
    return {
     'image': image,
     'image_name': image_name
    }

  def __len__(self) -> int:
    return len(self.name_to_relpath)
    pass


class CIRRDataset(Dataset):
  def __init__(self, split: str, preprocess: callable):
    self.preprocess = preprocess
    self.split = split

    if split not in ['test1', 'train', 'val']:
      raise ValueError("split should be in ['test1', 'train', 'val']")

    caption_file = f'cap.rc2.{split}.json'

    with open(os.path.join(base_path, 'cirr_dataset', 'cirr', 'captions', caption_file)) as f:
      self.triplets = json.load(f)

    with open(os.path.join(base_path, 'cirr_dataset', 'cirr', 'image_splits', f'split.rc2.{split}.json')) as f:
      self.name_to_relpath = json.load(f)

    LOGGER.info(f"CIRR {split} dataset initialized")

  def __getitem__(
    self,
    idx: int
  ) -> Dict[str, Union[str, torch.Tensor]]:
    try:
      group_members = self.triplets[idx]['img_set']['members']
      reference_name = self.triplets[idx]['reference']
      rel_caption = self.triplets[idx]['caption']

      if self.split == 'train':
        reference_image_path = os.path.join(base_path, 'cirr_dataset', self.name_to_relpath[reference_name])
        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
        target_hard_name = self.triplets[idx]['target_hard']
        target_image_path = os.path.join(base_path, 'cirr_dataset', self.name_to_relpath[target_hard_name])
        target_image = self.preprocess(PIL.Image.open(target_image_path))

        return {
          'reference_image': reference_image,
          'target_image': target_image,
          'caption': clip.tokenize(rel_caption, context_length=77, truncate=True)
        }

      elif self.split == 'val':
        reference_image_path = os.path.join(base_path, 'cirr_dataset', self.name_to_relpath[reference_name])
        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
        target_hard_name = self.triplets[idx]['target_hard']
        target_image_path = os.path.join(base_path, 'cirr_dataset', self.name_to_relpath[target_hard_name])
        target_image = self.preprocess(PIL.Image.open(target_image_path))
        return {
          'reference_image': reference_image,
          'target_image': target_image,
          'caption': clip.tokenize(rel_caption, context_length=77, truncate=True),
          'target_hard_name': target_hard_name,
          'group_member': group_members,
          'reference_name': reference_name,
        }

      elif self.split == 'test1':
        pair_id = self.triplets[idx]['pairid']
        return pair_id, reference_name, rel_caption, group_members
      else:
        raise NotImplementedError

    except Exception as e:
      LOGGER.info(f"Exception: {e}")

  def __len__(self) -> int:
    return len(self.triplets)


if __name__ == '__main__':
  from src.utils.transforms import targetpad_transform

  preprocess = targetpad_transform(1.25, 288)

  # train_dataset = CIRRDataset('train', preprocess)
  # train_dataloader = DataLoader(train_dataset, batch_size=64,
  #                               pin_memory=False, num_workers=8,
  #                               drop_last=True, shuffle=True)
  #
  # for atch in tqdm(train_dataloader):
  #   pass

  train_dataset = CIRRDataset('val', preprocess)
  train_dataloader = DataLoader(train_dataset, batch_size=64,
                                pin_memory=False, num_workers=8,
                                drop_last=True, shuffle=True)

  for atch in tqdm(train_dataloader):
    pass
