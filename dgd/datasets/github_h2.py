import os
# os.chdir("../")
import os.path as osp
import pathlib
from typing import Any, Sequence
import json
import pickle

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
import wandb


import concurrent.futures

from datetime import datetime

# import dgd.utils as utils
# from dgd.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
# from dgd.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
# from dgd.analysis.rdkit_functions import compute_molecular_metrics

import utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from analysis.rdkit_functions import compute_molecular_metrics
from analysis.visualization import TrainDiscreteNodeTypeVisualization, LargeGraphVisualization


from community_layout.layout_class import CommunityLayout

from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class GITH2Dataset(InMemoryDataset):
    raw_url = ('https://snap.stanford.edu/data/git_web_ml.zip')
    # raw_url2 = 'https://snap.stanford.edu/data/deezer_git_nets.zip'
    # processed_url = 'https://snap.stanford.edu/data/deezer_git_nets.zip'

    def __init__(self, stage, root, remove_h: bool, transform=None,
                 pre_transform=None, pre_filter=None, subsample = False,
                 max_size = 100, resolution = 20, n_samples = 120, n_workers = 4):
        print("\nStarting GIT dataset init\n")
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        self.subsample = subsample
        self.max_size = max_size
        self.resolution = resolution
        self.partitions = []
        self.n_samples = n_samples
        self.n_workers = n_workers
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['git_web_ml/musae_git_edges.json', 'git_web_ml/musae_git_partitions.pkl', 'git_web_ml/musae_git_target.csv', 'git_web_ml/musae_git_edges.csv']#, 'deezer_git_nets/deezer_edges.json']

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

        if self.subsample:
            sampler = MetropolisHastingsRandomWalkSampler(10000)
            G = sampler.sample(G)
            print(f"Loaded: {G}")
        print(f"Finding communities with a resolution of {self.resolution}")
        self.partitions = []
        sample_call_list = range(self.n_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_partition = {executor.submit(self.communities_split, G): i for i in sample_call_list}
            for future in concurrent.futures.as_completed(future_to_partition):
                partition = future.result()
                try:
                    self.partitions.append(partition)
                    # print(f"Success with partition {partition}")
                except:
                    print(f"Failed with partition {partition}")

        with open(self.raw_paths[1], "wb") as handle:
            pickle.dump(self.partitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.G = G
        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(self.G, handle, protocol=pickle.HIGHEST_PROTOCOL)


        #
        #
        #
        dataset = pd.DataFrame({'community_id':[i for i in range(len(self.partitions))]})#read_csv(self.raw_paths[1])
        # # dataset = dataset.sample(frac = 0.1)
        print(f"Done building CSV:\n{dataset.head()}")
        # # n_samples = len(dataset)
        n_train = int(0.8*self.n_samples)
        n_test = int(0.1 * self.n_samples)
        n_val = self.n_samples - (n_train + n_test)
        #
        # # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

        # quit()

    def communities_split(self, G):
        partition = comm.louvain_communities(G, resolution = self.resolution)

        self.community_diagnostics(G, partition)
        #
        # partition_dict = {}
        #
        # for i, p in enumerate(partition):
        #     subg = G.subgraph(p)
        #     edges = subg.edges()
        #     edges = [list(e) for e in edges]
        #     partition_dict[i] = edges
        #
        # del partition

        # self.partitions.append(partition)
        return partition
    def make_meta_graph(self, partition):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i:p for i,p in enumerate(partition)}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        self.partition = partition


        # Find unique community ids
        community_unique = set([k for k in community_to_node.keys()])

        # Produce a sub-graph for each community
        subgraphs = []
        for c in community_unique:
            subgraphs.append(nx.subgraph(self.G, community_to_node[c]))



        # Get nested list of edges in original graph
        G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(self.G)]

        # Build nested list of edges, of form [["community_id1", "community_id2"], ["community_id3", "community_id4"], ...]
        community_edgelist = []
        for e in G_edgelist:
            comm1 = partition[e[0]]
            comm2 = partition[e[1]]

            community_edgelist.append((comm1, comm2))

        # Find unique edges that are inter-community
        unique_comm_edges = list(set(community_edgelist))
        out_edges = []
        for e in unique_comm_edges:
            if (e[1], e[0]) not in out_edges and e[0] != e[1]:
                out_edges.append(e)
        unique_comm_edges = out_edges
        #
        # Count the number of times each inter-community edge occurs (and the inverse)
        # edge_count = [community_edgelist.count(e) + community_edgelist.count([e[1], e[0]]) for e in unique_comm_edges]
        #
        # Package inter-community edges and their counts as a list of tuples, [("community_id1", "community_id2", count),...]
        # full_description = [(*list(unique_comm_edges)[i], edge_count[i]) for i in range(len(edge_count))]


        # Build metagraph as a weighted networkx graph
        metaG = nx.Graph()
        # metaG.add_weighted_edges_from(full_description)
        metaG.add_edges_from(unique_comm_edges)


        # Set metagraph and community subgraphs as attributes
        # self.subgraphs = {i:g for i, g in enumerate(subgraphs)}
        return metaG

    def community_diagnostics(self, G, partition):

        # print(f"N communities: {len(partition)}")

        sizes = [len(partition[k]) for k in range(len(partition))]
        # print(f"Mean size: {np.mean(sizes)}\n"
        #       f"Num_communities: {len(sizes)}"
        #       f"Deviation: {np.std(sizes)}\n"
        #       f"Max size: {np.max(sizes)}\n"
        #       f"Min size: {np.min(sizes)}")

        wandb.log({"Mean_Community_Size": np.mean(sizes),
              "Community_Size_Deviation": np.std(sizes),
              "Num_communities": len(sizes),
              "Max_Community_Size": np.max(sizes),
              "Min_Community_Size": np.min(sizes)})




    def process(self):
        # RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2}#, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        print(target_df.head())
        
        node_type_df = pd.read_csv(self.raw_paths[2])
        unique_types = np.unique(node_type_df["ml_target"])
        types = {target:i for i, target in enumerate(unique_types.tolist())}

        # for f in open(self.raw_paths[0], "r"):
        #     all_edges = json.loads(f)
        # graphs = [nx.from_edgelist(all_edges[i]) for i in list(all_edges.keys())]
        with open(self.raw_paths[1], "rb") as handle:
            self.partitions = pickle.load(handle)
        with open(self.raw_paths[0], "rb") as handle:
            self.G = pickle.load(handle)

        graphs = [self.make_meta_graph(partition) for partition in self.partitions]


        densities = [nx.density(g) for g in graphs]
        wandb.log({"Mean_Density": np.mean(densities),
                   "Max_Density": np.max(densities),
                   "Min_Density": np.min(densities),
                   "Dev_Density": np.std(densities)})

        skip = []
        for i, G in enumerate(graphs):
            if G.number_of_nodes() > self.max_size:
                skip.append(i)

        suppl = tqdm(graphs)

        data_list = []

        all_nodes = []
        all_edges = []
        node_types = []
        edge_types = []

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

            this_partition = self.partitions[i]



            # typedict = {}
            type_idx = []
            for node in list(G.nodes()):
                community_nodes = this_partition[node]
                these_node_types = []
                for n in community_nodes:
                    these_node_types.append(node_type_df.at[n, "ml_target"])

                these_node_types, counts = np.unique(these_node_types, return_counts=True)
                if counts.shape[0] <= 1:
                    node_type = these_node_types[0]
                # elif np.abs(1 - (counts[0]/counts[1])) <= 0.1:
                #     node_type = 0
                else:
                    most_common_idx = np.argmax(counts)
                    node_type = these_node_types[most_common_idx]


                # node_type = node_type_df.at[node, "ml_target"]
                type_idx.append(types[node_type])
                # typedict[node] = node_type
                # all_nodes.append(types[node_type])

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            G = nx.convert_node_labels_to_integers(G)
            graphs_plotting.append(G)
            typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
            node_types.append(typedict)

            type_idx = []
            row, col, edge_type = [], [], []
            for edge in list(G.edges()):
                # print(G.edges[edge])
                start, end = edge[0], edge[1]
                row += [start, end]
                col += [end, start]
                etype = [1]

                edge_type += 2*etype

                type_idx.append(etype)

            typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
            edge_types.append(typedict)

            # all_edges += edge_type

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=4).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            # print(type_idx)
            # try:

            # except:
            #     continue
            y = torch.zeros((1, 0), dtype=torch.float)
            # values = target_df.loc[i]
            # target = values["ml_target"]
            # y = torch.Tensor([target]).reshape(1,1)
            # print(y)

            if self.remove_h:
                type_idx = torch.Tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                x = x[:, 1:]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        print(f"Data list: {data_list}")
        print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        # n_total = len(all_nodes)
        # type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
        # edge_type_counts = [all_edges.count(typ) for typ in list(np.unique(all_edges))]
        # # print(type_counts)
        # self.node_types = torch.tensor(type_counts) / n_total
        # self.edge_types
        # print(f"File node type marginals: {self.node_types}")

        visualization_tools = TrainDiscreteNodeTypeVisualization()

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path,
                                   f'graphs/train_communities/{self.stage}')
        visualization_tools.visualize(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                      node_types = node_types)
        visualization_tools.visualize_grid(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                      node_types=node_types, log = "real_grid")


        # LargeGraphVisualization(self.G.copy(), self.partition)




class GITH2DataModule(MolecularDataModule):
    def __init__(self, cfg):
        print("Entered GIT datamodule __init__")
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h
        print("Finished GIT datamodule __init__")

    def prepare_data(self) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': GITH2Dataset(stage='train',
                                        root=root_path,
                                        remove_h=self.cfg.dataset.remove_h,
                                        subsample = self.cfg.dataset.subsample,
                                        resolution = self.cfg.dataset.resolution,
                                        max_size = self.cfg.dataset.max_size,
                                        n_workers = self.cfg.train.num_workers),
                    'val': GITH2Dataset(stage='val',
                                      root=root_path,
                                      remove_h=self.cfg.dataset.remove_h,
                                      max_size = self.cfg.dataset.max_size,
                                        n_workers = self.cfg.train.num_workers),
                    'test': GITH2Dataset(stage='test',
                                       root=root_path,
                                       remove_h=self.cfg.dataset.remove_h,
                                       max_size = self.cfg.dataset.max_size,
                                        n_workers = self.cfg.train.num_workers)}
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
