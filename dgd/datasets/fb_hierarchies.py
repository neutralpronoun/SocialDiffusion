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
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
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


class FBHierarchiesDataset(InMemoryDataset):
    raw_url = ('https://snap.stanford.edu/data/facebook_large.zip')

    # raw_url2 = 'https://snap.stanford.edu/data/deezer_fb_nets.zip'
    # processed_url = 'https://snap.stanford.edu/data/deezer_fb_nets.zip'

    def __init__(self, stage, root, transform=None,
                 pre_transform=None, pre_filter=None, subsample=False, h = 1,
                 max_size=100, resolution=20, n_samples=120, n_workers=4):
        print("\nStarting FB dataset init\n")
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.h = h
        self.subsample = subsample
        self.max_size = max_size
        self.resolution = resolution
        self.partitions_h1 = {}
        self.partitions_h2 = []
        self.n_samples = n_samples
        self.n_workers = n_workers
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['facebook_large/musae_facebook_edges.json', 'facebook_large/musae_facebook_partitions.csv',
                'facebook_large/musae_facebook_target.csv',
                'facebook_large/musae_facebook_edges.csv']  # , 'deezer_fb_nets/deezer_edges.json']

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
        if self.h == 1:
            return ['proc_train_h1.pt', 'proc_val_h1.pt', 'proc_test_h1.pt']
        elif self.h == 2:
            return ['proc_train_h2.pt', 'proc_val_h2.pt', 'proc_test_h2.pt']
        elif self.h == 1.5:
            return ['proc_train_X.pt', 'proc_val_X.pt', 'proc_test_X.pt',
                    'proc_train_Y.pt', 'proc_val_Y.pt', 'proc_test_Y.pt']
        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")

    def download(self):
        """
        Download raw fb files. Taken from PyG FB class
        """

        print(self.raw_dir)
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            # file_path = download_url(self.raw_url2, self.raw_dir)
            # os.rename(osp.join(self.raw_dir, 'deezer_fb_nets/deezer_edges.json'),
            #           osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        edgelist = pd.read_csv(self.raw_paths[3])
        G = nx.from_pandas_edgelist(df=edgelist, source="id_1", target="id_2")
        del edgelist
        print(G)

        if self.subsample:
            sampler = MetropolisHastingsRandomWalkSampler(10000)
            G = sampler.sample(G)
            print(f"Loaded: {G}")

        print(f"\nFinding communities {self.n_samples} times with a resolution of {self.resolution}\n")
        self.partitions_h2 = []
        sample_call_list = range(self.n_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_partition = {executor.submit(self.communities_split, G): i for i in tqdm(sample_call_list)}
            for future in tqdm(concurrent.futures.as_completed(future_to_partition)):
                partition = future.result()
                try:
                    self.partitions_h2.append(partition)
                    # print(f"Success with partition {partition}")
                except:
                    print(f"Failed with partition {partition}")

        with open(self.raw_paths[1], "wb") as handle:
            pickle.dump(self.partitions_h2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.G = G
        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(self.G, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #
        #
        #
        dataset = pd.DataFrame(
            {'community_id': [i for i in range(len(self.partitions_h2))]})  # read_csv(self.raw_paths[1])
        # # dataset = dataset.sample(frac = 0.1)
        print(f"Done building CSV:\n{dataset.head()}")
        # # n_samples = len(dataset)
        n_train = int(0.8 * self.n_samples)
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
        partition = comm.louvain_communities(G, resolution=self.resolution)

        self.community_diagnostics(partition)
        return partition

    def communities_to_edges(self):
        partition_dict = {}
        for run in self.partitions_h2:
            for i, p in enumerate(run):
                subg = self.G.subgraph(p)
                edges = subg.edges()
                edges = [list(e) for e in edges]
                partition_dict[i] = edges
            # del partition_dict
            if self.partitions_h1 == {}:
                max_partition_so_far = 0
            else:
                max_partition_so_far = max(list(self.partitions_h1.keys()))

            for p in partition_dict:
                self.partitions_h1[p + max_partition_so_far] = partition_dict[p]
        print(p, max_partition_so_far, p + max_partition_so_far)

    def make_meta_graph(self, partition):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i: p for i, p in enumerate(partition)}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        # self.partition = partition

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


    def get_joined_pairs(self, partition):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i: p for i, p in enumerate(partition)}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        # self.partition = partition

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

        return unique_comm_edges

    def community_diagnostics(self, partition):

        # print(f"N communities: {len(partition)}")

        sizes = [len(partition[k]) for k in range(len(partition))]

        wandb.log({"Mean_Community_Size": np.mean(sizes),
                   "Community_Size_Deviation": np.std(sizes),
                   "Num_communities": len(sizes),
                   "Max_Community_Size": np.max(sizes),
                   "Min_Community_Size": np.min(sizes)})

    def process(self):
        # RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2}  # , 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        print(target_df.head())

        node_type_df = pd.read_csv(self.raw_paths[2])
        unique_types = np.unique(node_type_df["page_type"])
        types = {target: i for i, target in enumerate(unique_types.tolist())}

        # for f in open(self.raw_paths[0], "r"):
        #     all_edges = json.loads(f)
        # graphs_h2 = [nx.from_edgelist(all_edges[i]) for i in list(all_edges.keys())]
        with open(self.raw_paths[1], "rb") as handle:
            self.partitions_h2 = pickle.load(handle)
        with open(self.raw_paths[0], "rb") as handle:
            self.G = pickle.load(handle)
        

        if self.h == 1:

            self.communities_to_edges()

            graphs_h1 = [nx.from_edgelist(self.partitions_h1[i]) for i in list(self.partitions_h1.keys())]

            skip = []
            for i, G in enumerate(graphs_h1):
                if G.number_of_nodes() > self.max_size:
                    skip.append(i)

            suppl = tqdm(graphs_h1)

            data_list = []

            all_nodes = []
            node_types = []

            node_types = []
            edge_types = []
            graphs_h1_plotting = []

            for i, G in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                try:
                    nodelist = list(G.nodes())
                    N = G.number_of_nodes()
                    min_node = min(nodelist)
                except:
                    continue



                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])



                G = nx.convert_node_labels_to_integers(G)
                # graphs[i] = G
                graphs_h1_plotting.append(G)
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


                # y = torch.zeros((1, 0), dtype=torch.float)
                these_node_types, counts = np.unique(type_idx, return_counts=True)
                if counts.shape[0] <= 1:
                    graph_type = these_node_types[0]
                else:
                    most_common_idx = np.argmax(counts)
                    graph_type = these_node_types[most_common_idx]
                # print(these_node_types, counts, graph_type)

                # y = torch.Tensor([graph_type]).reshape(1,1)
                y = F.one_hot(torch.tensor([graph_type]), num_classes=len(types)).float()
                # print("\n")
                # print(y)
                # print(x)
                # values = target_df.loc[i]
                # target = values["target"]
                # y = torch.Tensor([target]).reshape(1,1)
                # print(y)

                # if self.remove_h:
                #     type_idx = torch.Tensor(type_idx).long()
                #     to_keep = type_idx > 0
                #     # print(f"To keep {to_keep}")
                #     # print(f"Edge index/attr: {edge_index} {edge_attr}")
                #     edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                #                                      num_nodes=len(to_keep))
                #     # print(f"Edge index/attr: {edge_index} {edge_attr}")
                #     # print(f"X: {x}")
                #     x = x[to_keep]
                #     # Shift onehot encoding to match atom decoder
                #     x = x[:, 1:]
                #     #
                #     # print(f"X: {x}")
                #     # quit()

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

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h1')
                visualization_tools.visualize(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                              node_types = node_types)
                visualization_tools.visualize_grid(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                              node_types=node_types, log = "h1_real")

        elif self.h == 1.5:

            self.communities_to_edges()

            X_graphs = []
            Y_graphs = []

            if self.stage == 0:
                n_runs_considered = 2
            else:
                n_runs_considered = 1

            for model_partition in tqdm(self.partitions_h2[:n_runs_considered]):
                joined_pairs = self.get_joined_pairs(model_partition)
                for pair in tqdm(joined_pairs):
                    x1, x2 = pair[0], pair[1]


                    nodes1, nodes2 = model_partition[x1], model_partition[x2]

                    if len(nodes1) > self.max_size or len(nodes2) > self.max_size:
                        continue

                    g1, g2, g3 = nx.subgraph(self.G, nodes1), nx.subgraph(self.G, nodes2), nx.subgraph(self.G, nodes1.union(nodes2))

                    separate_graph = nx.compose(g1, g2)
                    X_graphs.append(separate_graph)
                    Y_graphs.append(g3)

            skip = []
            for i, G in enumerate(X_graphs):
                if G.number_of_nodes() > 2 * self.max_size:
                    skip.append(i)
            data_list = []
            all_nodes = []
            node_types = []
            graphs_plotting = []

            suppl = tqdm(X_graphs)

            for i, G in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                try:
                    nodelist = list(G.nodes())
                    N = G.number_of_nodes()
                    min_node = min(nodelist)
                except:
                    continue



                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])



                G = nx.convert_node_labels_to_integers(G)
                # graphs[i] = G
                graphs_plotting.append(G)
                typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
                # print(typedict)
                node_types.append(typedict)

                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [1]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)

                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)


                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                these_node_types, counts = np.unique(type_idx, return_counts=True)
                if counts.shape[0] <= 1:
                    graph_type = these_node_types[0]
                else:
                    most_common_idx = np.argmax(counts)
                    graph_type = these_node_types[most_common_idx]

                y = F.one_hot(torch.tensor([graph_type]), num_classes=len(types)).float()


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

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_X')
                visualization_tools.visualize(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types = node_types, largest_component = False)
                visualization_tools.visualize_grid(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types=node_types, log = "Community_Pairs_X", largest_component = False)

            data_list = []
            all_nodes = []
            node_types = []
            graphs_plotting = []

            suppl = tqdm(Y_graphs)

            for i, G in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                try:
                    nodelist = list(G.nodes())
                    N = G.number_of_nodes()
                    min_node = min(nodelist)
                except:
                    continue



                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])



                G = nx.convert_node_labels_to_integers(G)
                # graphs[i] = G
                graphs_plotting.append(G)
                typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
                # print(typedict)
                node_types.append(typedict)

                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [1]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)

                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)


                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                these_node_types, counts = np.unique(type_idx, return_counts=True)
                if counts.shape[0] <= 1:
                    graph_type = these_node_types[0]
                else:
                    most_common_idx = np.argmax(counts)
                    graph_type = these_node_types[most_common_idx]

                y = F.one_hot(torch.tensor([graph_type]), num_classes=len(types)).float()


                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx + 3])

            n_total = len(all_nodes)
            type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
            # print(type_counts)
            self.node_types = torch.tensor(type_counts) / n_total
            print(f"File node type marginals: {self.node_types}")

            visualization_tools = TrainDiscreteNodeTypeVisualization()

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_Y')
                visualization_tools.visualize(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types = node_types)
                visualization_tools.visualize_grid(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types=node_types, log = "Community_Pairs_Y", largest_component = False)



        elif self.h == 2:

            graphs_h2 = [self.make_meta_graph(partition) for partition in self.partitions_h2]

            densities = [nx.density(g) for g in graphs_h2]
            wandb.log({"Mean_Density": np.mean(densities),
                       "Max_Density": np.max(densities),
                       "Min_Density": np.min(densities),
                       "Dev_Density": np.std(densities)})

            skip = []
            for i, G in enumerate(graphs_h2):
                if G.number_of_nodes() > self.max_size:
                    skip.append(i)

            suppl = tqdm(graphs_h2)

            data_list = []

            node_types = []
            edge_types = []
            graphs_h2_plotting = []

            for i, G in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                try:
                    nodelist = list(G.nodes())
                    N = G.number_of_nodes()
                    min_node = min(nodelist)
                except:
                    continue

                this_partition = self.partitions_h2[i]

                # typedict = {}
                type_idx = []
                for node in list(G.nodes()):
                    community_nodes = this_partition[node]
                    these_node_types = []
                    for n in community_nodes:
                        these_node_types.append(node_type_df.at[n, "page_type"])

                    these_node_types, counts = np.unique(these_node_types, return_counts=True)
                    if counts.shape[0] <= 1:
                        node_type = these_node_types[0]
                    else:
                        most_common_idx = np.argmax(counts)
                        node_type = these_node_types[most_common_idx]

                    type_idx.append(types[node_type])

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                G = nx.convert_node_labels_to_integers(G)
                graphs_h2_plotting.append(G)
                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                node_types.append(typedict)

                type_idx = []
                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    etype = [1]

                    edge_type += 2 * etype

                    type_idx.append(etype)

                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                edge_types.append(typedict)

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                y = torch.zeros((1, 0), dtype=torch.float)

                # if self.remove_h:
                #     type_idx = torch.Tensor(type_idx).long()
                #     to_keep = type_idx > 0
                #     edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                #                                      num_nodes=len(to_keep))
                #     x = x[to_keep]
                #     x = x[:, 1:]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            print(f"Data list: {data_list}")
            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            visualization_tools = TrainDiscreteNodeTypeVisualization()

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h1')
                visualization_tools.visualize(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                              node_types = node_types)
                visualization_tools.visualize_grid(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                              node_types=node_types, log = "h2_real")





        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")




class FBHierarchiesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        print("Entered FB datamodule __init__")
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.h = cfg.dataset.h
        print("Finished FB datamodule __init__")

    def prepare_data(self) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': FBHierarchiesDataset(stage='train',
                                         root = root_path,
                                         h = self.h,
                                         subsample=self.cfg.dataset.subsample,
                                         resolution=self.cfg.dataset.resolution,
                                         max_size=self.cfg.dataset.max_size,
                                         n_workers=self.cfg.train.num_workers),
                    'val': FBHierarchiesDataset(stage='val',
                                       root=root_path,
                                       h = self.h,
                                       max_size=self.cfg.dataset.max_size,
                                       n_workers=self.cfg.train.num_workers),
                    'test': FBHierarchiesDataset(stage='test',
                                        root=root_path,
                                        h=self.h,
                                        max_size=self.cfg.dataset.max_size,
                                        n_workers=self.cfg.train.num_workers)}
        super().prepare_data(datasets)


class FBDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = dataset_config.name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = datamodule.node_types()  # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
