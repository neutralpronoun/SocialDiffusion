import os
# os.chdir("../")
import os.path as osp
import pathlib
from typing import Any, Sequence
import json

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import networkx as nx
import networkx.algorithms.community as comm

from datetime import datetime

# import dgd.utils as utils
# from dgd.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
# from dgd.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
# from dgd.analysis.rdkit_functions import compute_molecular_metrics

import utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from analysis.rdkit_functions import compute_molecular_metrics
from analysis.visualization import TrainDiscreteNodeTypeVisualization


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class GITDataset(InMemoryDataset):
    raw_url = ('https://snap.stanford.edu/data/git_web_ml.zip')
    # raw_url2 = 'https://snap.stanford.edu/data/deezer_git_nets.zip'
    # processed_url = 'https://snap.stanford.edu/data/deezer_git_nets.zip'

    def __init__(self, stage, root, remove_h: bool, transform=None, pre_transform=None, pre_filter=None):
        print("\nStarting GIT dataset init\n")
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['git_web_ml/musae_git_edges.json', 'git_web_ml/musae_git_partitions.csv', 'git_web_ml/musae_git_target.csv', 'git_web_ml/musae_git_edges.csv']#, 'deezer_git_nets/deezer_edges.json']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):
        """
        Download raw git files. Taken from PyG GIT class
        """

        print(self.raw_dir)
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            # file_path = download_url(self.raw_url2, self.raw_dir)
            # os.rename(osp.join(self.raw_dir, 'deezer_git_nets/deezer_edges.json'),
            #           osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return


        edgelist = pd.read_csv(self.raw_paths[3])
        G = nx.from_pandas_edgelist(df = edgelist, source="id_1", target="id_2")
        del edgelist
        print(G)
        self.communities_split(G)

        del G



        dataset = pd.read_csv(self.raw_paths[1])
        # dataset = dataset.sample(frac = 0.1)
        print(f"Done building CSV:\n{dataset.head()}")
        n_samples = len(dataset)
        n_train = int(0.8*n_samples)
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

        # quit()

    def communities_split(self, G):
        partition = comm.louvain_communities(G, resolution = 10)
        # partition_dict = {i:list(partition[i]) for i in range(len(partition))}
        # self.raw_paths[0] = 'github_large/musae_github_edges.json'

        partition_dict = {}

        for i, p in enumerate(partition):
            subg = G.subgraph(p)
            edges = subg.edges()
            edges = [list(e) for e in edges]
            partition_dict[i] = edges

        with open(self.raw_paths[0], 'w') as f:
            json.dump(partition_dict, f)

        partition_df = pd.DataFrame({'community_id':[i for i in range(len(partition))]})
        partition_df.to_csv(self.raw_paths[1])

        del partition
        del partition_dict
        del partition_df

    def process(self):
        # RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2}#, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        print(target_df.head())
        
        node_type_df = pd.read_csv(self.raw_paths[2])
        unique_types = np.unique(node_type_df["ml_target"])
        # print(unique_types)
        types = {target:i for i, target in enumerate(unique_types.tolist())}
        # print(types)

        for f in open(self.raw_paths[0], "r"):
            all_edges = json.loads(f)
        graphs = [nx.from_edgelist(all_edges[i]) for i in list(all_edges.keys())]

        skip = []
        for i, G in enumerate(graphs):
            if G.number_of_nodes() > 100:
                skip.append(i)

        suppl = tqdm(graphs)

        data_list = []

        all_nodes = []
        node_types = []

        graphs_plotting = []

        for i, G in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            try:
                nodelist = list(G.nodes())
                N = G.number_of_nodes()
                min_node = min(nodelist)
            except:
                continue

            # typedict = {}
            type_idx = []
            for node in list(G.nodes()):
                node_type = node_type_df.at[node, "ml_target"]
                type_idx.append(types[node_type])
                # typedict[node] = node_type



            G = nx.convert_node_labels_to_integers(G)
            # graphs[i] = G
            graphs_plotting.append(G)
            typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
            # print(typedict)
            node_types.append(typedict)

            row, col, edge_type = [], [], []
            for edge in list(G.edges()):
                start, end = edge[0], edge[1]#bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [1]#[bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            # print(edge_type)
            edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)
            # print(edge_attr)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            # print(type_idx)
            try:
                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            except:
                continue
            y = torch.zeros((1, 0), dtype=torch.float)
            # values = target_df.loc[i]
            # target = values["ml_target"]
            # y = torch.Tensor([target]).reshape(1,1)
            # print(y)

            if self.remove_h:
                type_idx = torch.Tensor(type_idx).long()
                to_keep = type_idx > 0
                # print(f"To keep {to_keep}")
                # print(f"Edge index/attr: {edge_index} {edge_attr}")
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                # print(f"Edge index/attr: {edge_index} {edge_attr}")
                # print(f"X: {x}")
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]
                #
                # print(f"X: {x}")
                # quit()

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        n_total = len(all_nodes)
        type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
        # print(type_counts)
        self.node_types = torch.tensor(type_counts) / n_total
        print(f"File node type marginals: {self.node_types}")

        visualization_tools = TrainDiscreteNodeTypeVisualization()

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path,
                                   f'graphs/train_communities/{self.stage}')
        visualization_tools.visualize(result_path, graphs_plotting, min(15, len(graphs_plotting)), node_types = node_types)


class GITDataModule(MolecularDataModule):
    def __init__(self, cfg):
        print("Entered GIT datamodule __init__")
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h
        print("Finished GIT datamodule __init__")

    def prepare_data(self) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': GITDataset(stage='train', root=root_path, remove_h=self.cfg.dataset.remove_h),
                    'val': GITDataset(stage='val', root=root_path, remove_h=self.cfg.dataset.remove_h),
                    'test': GITDataset(stage='test', root=root_path, remove_h=self.cfg.dataset.remove_h)}
        super().prepare_data(datasets)


class GITDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = dataset_config.name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = datamodule.node_types()               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

#
# class GITinfos(AbstractDatasetInfos):
#     def __init__(self, datamodule, cfg, recompute_statistics=False):
#         self.remove_h = cfg.dataset.remove_h
#         self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
#
#         self.name = 'git'
#         if self.remove_h:
#             self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
#             self.atom_decoder = ['C', 'N', 'O', 'F']
#             self.num_atom_types = 4
#             self.valencies = [4, 3, 2, 1]
#             self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
#             self.max_n_nodes = 9
#             self.max_weight = 150
#             self.n_nodes = torch.Tensor([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
#                                          0.0046472, 0.023985, 0.13666, 0.83337])
#             self.node_types = torch.Tensor([0.7230, 0.1151, 0.1593, 0.0026])
#             self.edge_types = torch.Tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])
#
#             super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
#             self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
#             self.valency_distribution[0: 6] = torch.Tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])
#         else:
#             self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
#             self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
#             self.valencies = [1, 4, 3, 2, 1]
#             self.num_atom_types = 5
#             self.max_n_nodes = 29
#             self.max_weight = 390
#             self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
#             self.n_nodes = torch.Tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
#                                          9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
#                                          1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
#                                          1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
#                                          5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])
#
#             self.node_types = torch.Tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
#             self.edge_types = torch.Tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])
#
#             super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
#             self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
#             self.valency_distribution[0:6] = torch.Tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])
#
#         if recompute_statistics:
#             np.set_printoptions(suppress=True, precision=5)
#             self.n_nodes = datamodule.node_counts()
#             print("Distribution of number of nodes", self.n_nodes)
#             np.savetxt('n_counts.txt', self.n_nodes.numpy())
#             self.node_types = datamodule.node_types()                                     # There are no node types
#             print("Distribution of node types", self.node_types)
#             np.savetxt('atom_types.txt', self.node_types.numpy())
#
#             self.edge_types = datamodule.edge_counts()
#             print("Distribution of edge types", self.edge_types)
#             np.savetxt('edge_types.txt', self.edge_types.numpy())
#
#             valencies = datamodule.valency_count(self.max_n_nodes)
#             print("Distribution of the valencies", valencies)
#             np.savetxt('valencies.txt', valencies.numpy())
#             self.valency_distribution = valencies
#             assert False

#
# def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False, ):
#     if evaluate_dataset:
#         assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
#     datadir = cfg.dataset.datadir
#     remove_h = cfg.dataset.remove_h
#     atom_decoder = dataset_infos.atom_decoder
#     root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
#     smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
#     smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
#     if os.path.exists(smiles_path):
#         print("Dataset smiles were found.")
#         train_smiles = np.load(smiles_path)
#     else:
#         print("Computing dataset smiles...")
#         train_smiles = compute_git_smiles(atom_decoder, train_dataloader, remove_h)
#         np.save(smiles_path, np.array(train_smiles))
#
#     if evaluate_dataset:
#         train_dataloader = train_dataloader
#         all_molecules = []
#         for i, data in enumerate(train_dataloader):
#             dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
#             dense_data = dense_data.mask(node_mask, collapse=True)
#             X, E = dense_data.X, dense_data.E
#
#             for k in range(X.size(0)):
#                 n = int(torch.sum((X != -1)[k, :]))
#                 atom_types = X[k, :n].cpu()
#                 edge_types = E[k, :n, :n].cpu()
#                 all_molecules.append([atom_types, edge_types])
#
#         print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
#         metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
#                                             dataset_info=dataset_infos)
#         print(metrics[0])
#
#     return train_smiles
#
#
# def compute_git_smiles(atom_decoder, train_dataloader, remove_h):
#     '''
#
#     :param dataset_name: git or git_second_half
#     :return:
#     '''
#     print(f"\tConverting GIT dataset to SMILES for remove_h={remove_h}...")
#
#     mols_smiles = []
#     len_train = len(train_dataloader)
#     invalid = 0
#     disconnected = 0
#     for i, data in enumerate(train_dataloader):
#         dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
#         dense_data = dense_data.mask(node_mask, collapse=True)
#         X, E = dense_data.X, dense_data.E
#
#         n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]
#
#         molecule_list = []
#         for k in range(X.size(0)):
#             n = n_nodes[k]
#             atom_types = X[k, :n].cpu()
#             edge_types = E[k, :n, :n].cpu()
#             molecule_list.append([atom_types, edge_types])
#
#         for l, molecule in enumerate(molecule_list):
#             mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
#             smile = mol2smiles(mol)
#             if smile is not None:
#                 mols_smiles.append(smile)
#                 mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
#                 if len(mol_frags) > 1:
#                     print("Disconnected molecule", mol, mol_frags)
#                     disconnected += 1
#             else:
#                 print("Invalid molecule obtained.")
#                 invalid += 1
#
#         if i % 1000 == 0:
#             print("\tConverting GIT dataset to SMILES {0:.2%}".format(float(i) / len_train))
#     print("Number of invalid molecules", invalid)
#     print("Number of disconnected molecules", disconnected)
#     return mols_smiles
