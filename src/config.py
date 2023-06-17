import json
from dataclasses import dataclass
from typing import Optional

from src.utils.constants import LOGGER


class BaseConfig:
  def save_config(self, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
      LOGGER.info(f'Experiment config will be saved at {path}')
      json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

  @classmethod
  def from_json_file(cls, path: str):
    with open(path, 'r', encoding='utf-8') as f:
      LOGGER.info(f'Loading Experiment config: {path}')
      json_dict = json.load(f)
    return cls(**json_dict)


@dataclass(frozen=True)
class BaselineConfig(BaseConfig):
  # CLIP
  logit_scale_init_value: Optional[float] = 2.6592
  base_path: Optional[str] = '/opt/home/jude/workspace/clip_retrieval/cirr_dataset/'
  clip_model_name: Optional[str] = 'RN50x4'
