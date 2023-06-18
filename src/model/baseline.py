import logging
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BaseConfig, BaselineConfig
from src.model.base import Base
from src.model.modules.opts import element_wise_sum
from src.third_party.clip.model import CLIP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel(Base):
  def __init__(
    self,
    config: Union[BaseConfig, BaselineConfig],
    backbone: Union[CLIP, nn.Module]
  ) -> None:
    super().__init__()
    self.backbone: Union[CLIP, nn.Module] = backbone
    self.loss_fn = nn.CrossEntropyLoss()
    self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

  def forward(
    self,
    input_ids: torch.Tensor,
    reference_feats: torch.Tensor,
    target_feats: torch.Tensor,
  ) -> Dict[str, torch.Tensor]:
    text_features = self.generate_embedding(input_ids, 'text')
    reference_features = self.generate_embedding(reference_feats, embed_type='image')
    target_features = F.normalize(self.generate_embedding(target_feats, embed_type='image'))

    predicted_features = element_wise_sum(reference_features, text_features)
    logits = self.logit_scale * predicted_features @ target_features.T
    images_in_batch = reference_features.size(0)

    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=text_features.device)
    loss = self.loss_fn(logits, ground_truth)
    return {
      'loss': loss
    }

  def generate_embedding(
    self,
    input_tensor: torch.Tensor,
    embed_type: str
  ) -> torch.Tensor:
    if embed_type == 'text':
      return self.backbone.encode_text(input_tensor)
    elif embed_type == 'image':
      return self.backbone.encode_image(input_tensor)
    else:
      raise ValueError(f"This doesn't support embed_type: {embed_type}")

  def fetch_top_k_from_datastore(self):
    pass


