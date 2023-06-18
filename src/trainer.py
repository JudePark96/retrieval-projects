from abc import ABC, abstractmethod
from typing import Union, Dict

import torch

from src.config import BaseConfig, BaselineConfig


class BaseTrainer(ABC):
  def __init__(
    self,
    config: Union[BaseConfig, BaselineConfig]
  ) -> None:
    super().__init__()

  # def load_clip_model(self):
  #   self.clip, self.preprocess = clip.load(self.config.clip_model_name, device=self.device, jit=False)

  @abstractmethod
  def register_external_logging_tools(self):
    pass

  @abstractmethod
  def register_dataloader(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def register_model(self) -> None:
    raise NotImplementedError

  @abstractmethod
  def register_optimizer(self):
    raise NotImplementedError

  @abstractmethod
  def train(self):
    raise NotImplementedError

  @abstractmethod
  def training_epoch(self, epoch: int):
    raise NotImplementedError

  @abstractmethod
  def training_step(self, step: int, batch: Dict[str, Union[str, torch.Tensor]]):
    raise NotImplementedError

  @abstractmethod
  @torch.no_grad()
  def eval(self):
    raise NotImplementedError
