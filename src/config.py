import json
from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass()
class BaselineConfig(BaseConfig):
  # wandb
  project_name: str = 'llm-cirr-retrieval'
  logging_per_step: int = 10

  seed: Optional[int] = 13
  model_type: Optional[str] = 'baseline'
  preprocess_func: Optional[str] = 'targetpad'  # [squarepad, targetpad, clip]
  torch_dtype: Optional[int] = 'fp16'
  run_predict: Optional[bool] = False

  learning_rate: Optional[float] = 2e-5
  adam_betas: Optional[Tuple[int]] = (0.9, 0.999)
  adam_epsilon: Optional[float] = 1e-7
  weight_decay: Optional[float] = 0.01
  warmup_proportion: Optional[float] = 0.06

  num_train_epochs: Optional[int] = 30
  gradient_accumulation_steps: Optional[int] = 1

  index_batch_size: Optional[int] = 16  # This is the batch size for indexing embedding.
  train_batch_size_per_gpu: Optional[int] = 64
  eval_batch_size_per_gpu: Optional[int] = 16

  max_grad_norm: Optional[int] = 1.0

  # CLIP
  use_clip: Optional[bool] = True
  logit_scale_init_value: Optional[float] = 2.6592
  base_path: Optional[str] = '/opt/home/jude/workspace/clip_retrieval/'
  clip_model_name: Optional[str] = 'RN50x4'
  clip_input_dim: Optional[int] = 288
