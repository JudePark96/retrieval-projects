import os
import json
import random

import numpy as np
import torch


def print_args(cfg, logger):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in cfg.items():
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s =   %s", keystr, val)
    logger.info(" *********************************************** ")


def save_configuration(configuration, file_path):
    with open(file_path, 'w') as wf:
        json.dump(configuration, wf, indent=2, ensure_ascii=False)


def load_configuration(file_path):
    with open(file_path, 'r') as rf:
        configuration = json.load(rf)
    return configuration


def sanity_checks(cfg, output_dir_empty=True):
    if not os.path.exists(cfg['source_data_dir']):
        raise FileNotFoundError(f"don't exist the source dataset dir: {cfg['source_data_dir']}")

    if os.path.isdir(cfg['output_dir']) and output_dir_empty:
        assert not os.listdir(cfg['output_dir']), "directory must be empty!! (change argument 'must_emtpy=False')"
    else:
        os.makedirs(cfg['output_dir'], exist_ok=True)


def read_json_lines(data_path):
    with open(data_path, "r") as rf:
        instances = list(map(lambda x: json.loads(x), rf.readlines()))
    return instances


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
