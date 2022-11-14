# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import sys

# sys.path.insert(1, 'dgd')

import pathlib
import warnings

from datetime import datetime


import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# from dgd import utils
# from dgd.datasets import guacamol_dataset, qm9_dataset, moses_dataset
# from dgd.datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
# from dgd.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
# from dgd.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
# from diffusion_model import LiftedDenoisingDiffusion
# from diffusion_model_discrete import DiscreteDenoisingDiffusion
# from dgd.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics, \
#     TrainMolecularMetricsDiscrete
# from dgd.analysis.visualization import MolecularVisualization, NonMolecularVisualization
# from dgd.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
# from dgd.diffusion.extra_features_molecular import ExtraMolecularFeatures

import utils



# from datasets import guacamol_dataset, qm9_dataset#, moses_dataset
# from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos

from datasets import guacamol_dataset, qm9_dataset#, moses_dataset
# from datasets import ego_dataset
from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
from datasets import ego_dataset, fb_dataset


from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, EGOSamplingMetrics, FBSamplingMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics, \
    TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization, NonMolecularVisualization, DiscreteNodeTypeVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg):
    # now =
    # datetime_str =
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../configs', config_name='config') #
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print("\nEntered main\n")
    print("\n")
    print(cfg)
    print("\n")

    print("\nUtils and CFG line\n")
    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)


    if dataset_config["name"] in ["ego", "fb"]:
        if dataset_config["name"] == "ego":
            print("\nRecognised ego dataset\n")
            datamodule = ego_dataset.EGODataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = EGOSamplingMetrics(datamodule.dataloaders)
            dataset_infos = ego_dataset.EGODatasetInfos(datamodule,
                                                        dataset_config)  # SpectreDatasetInfos(datamodule, dataset_config)

        elif dataset_config["name"] == "fb":
            print("\nRecognised ego dataset\n")
            datamodule = fb_dataset.FBDataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = FBSamplingMetrics(datamodule.dataloaders)
            dataset_infos = fb_dataset.FBDatasetInfos(datamodule,
                                                        dataset_config)  # SpectreDatasetInfos(datamodule, dataset_config)
        print(f"Metric infos etc")

        print(dataset_infos)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = DiscreteNodeTypeVisualization()

        print(f"Extra features")
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        print(f"Computing in/out dims")
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['sbm', 'comm-20', 'planar']:
        if dataset_config['name'] == 'sbm':
            datamodule = SBMDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
        elif dataset_config['name'] == 'comm-20':
            datamodule = Comm20DataModule(cfg)
            sampling_metrics = Comm20SamplingMetrics(datamodule.dataloaders)

        else:
            datamodule = PlanarDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule.dataloaders)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        if dataset_config["name"] == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            datamodule.prepare_data()
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            print("\nRecognised guacamol dataset\n")
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None

        elif dataset_config.name == 'moses':
            datamodule = moses_dataset.MOSESDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None

        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))


    print("\nFinished building datasets!\n")
    print(f"Output from test resume:\n{cfg.general}")
    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])



    print("\nGetting model\n")
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    print("\nGetting callbacks\n")
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'test':
        print("[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    print(f"\nGetting trainer, GPU status: {'gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'}\n")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks,
                      num_sanity_val_steps=0,
                      logger=[])

    if not cfg.general.test_only:
        print("\nEntering trainer\n")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        print("\nEntering tester\n")
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    print("Get past imports")
    main()
