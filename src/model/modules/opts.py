import logging

import torch
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def element_wise_sum(
  image_features: torch.tensor,
  text_features: torch.tensor
) -> torch.tensor:
  return F.normalize(image_features + text_features, dim=-1)
