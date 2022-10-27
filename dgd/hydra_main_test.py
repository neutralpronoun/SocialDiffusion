try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import sys

# sys.path.insert(1, 'dgd')

import pathlib
import warnings

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

import torch_geometric
import torch_sparse
import torch_cluster

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

print(f"\n Non-DiGress packages import OK\n")


@hydra.main(version_base='1.1', config_path='../configs', config_name='config') #
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print("\nEntered main\n")
    print(f"Dataset config: {dataset_config}")

if __name__ == "__main__":

    main()