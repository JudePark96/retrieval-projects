import logging
from abc import ABC, ABCMeta, abstractmethod

import torch
from torch import nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Base(ABC, nn.Module):
  def __init__(self) -> None:
    super().__init__()

  @abstractmethod
  def generate_embedding(
    self,
    input_tensor: torch.Tensor,
    embed_type: str
  ) -> torch.Tensor:
    raise NotImplementedError()

  @abstractmethod
  def fetch_top_k_from_datastore(self):
    raise NotImplementedError()

