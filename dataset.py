import pickle as pkl
import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
from tqdm import tqdm


class MyDataset(InMemoryDataset):
    def __init__(self, root='./processed_data/nyc', set='train', percentage=0.1, transform=None, pre_transform=None):
        # set is 'train' or 'test' or 'val'
        self.set = set

        # the percentage of data to be used
        self.percentage=percentage
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # the raw data file generated by preprocess.py
        return [self.set + '.pkl']

    @property
    def processed_file_names(self):
        # the file to save the processed data
        return [self.set + '_' + str(self.percentage) + '_seq_graph.pt']

    def download(self):
        # no need to download
        pass

    def process(self):
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'rb') as f:
            data = pkl.load(f)
            print(f'orignial data num: {len(data)}')
            
        data_list = []
        for uid, poi, seq, coord, y in tqdm(data[:int(len(data)*self.percentage)]):
            # the first appearance order of the poi in the sequence
            idx = 0
            x = []
            node2idx = dict()
            for node in seq:
                if node not in node2idx:
                    node2idx[node] = idx
                    x.append([node])
                    idx += 1
            idx_seq = [node2idx[node] for node in seq]
            # x is the poi of each node, size (num_nodes, 1)
            x = torch.LongTensor(x)
            # edge_index is the edge of the graph, size (2, num_edges)
            edge_index = torch.LongTensor([idx_seq[:-1], idx_seq[1:]])
            # y is the label (0 or 1) of the sample, size (1)
            y = torch.LongTensor([y])
            # size (1)
            uid = torch.LongTensor([uid])
            # target poi, size (1)
            poi = torch.LongTensor([poi])
            # coordinate of target poi, size (2)
            coord = torch.Tensor(coord)

            data_list.append(Data(x=x, edge_index=edge_index,
                             y=y, uid=uid, poi=poi, coord=coord))

        self.save(data_list, self.processed_paths[0])
