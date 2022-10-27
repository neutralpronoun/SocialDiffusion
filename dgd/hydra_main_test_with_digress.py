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


print("Starting DiGress files import")

"""Comment this out to check that non-digress files packages are playing nicely (excluding graph-tool and rdkit)"""

import utils
print("import 1")
from datasets import guacamol_dataset
print("import 2")
from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
print("import 3")
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
print("import 4")
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
print("import 5")
from diffusion_model import LiftedDenoisingDiffusion
print("import 6")
from diffusion_model_discrete import DiscreteDenoisingDiffusion
print("import 7")
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics, \
    TrainMolecularMetricsDiscrete
print("import 8")
from analysis.visualization import MolecularVisualization, NonMolecularVisualization
print("import 9")
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
print("import 10")
from diffusion.extra_features_molecular import ExtraMolecularFeatures
print("import 11")

@hydra.main(version_base='1.1', config_path='../configs', config_name='config') #
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print("\nEntered main\n")
    print(f"Dataset config: {dataset_config}")

if __name__ == "__main__":

    main()