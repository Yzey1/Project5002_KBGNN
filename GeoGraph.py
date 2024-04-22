import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree


class SelfAttn(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(SelfAttn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True)

    def forward(self, sess_embed, sections):
        v_i = torch.split(sess_embed, sections)
        v_i_pad = pad_sequence(v_i, batch_first=True, padding_value=0.)
        # v_i = torch.stack(v_i)

        attn_output, _ = self.multihead_attn(v_i_pad, v_i_pad, v_i_pad)

        return attn_output


class Geo_GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) module for processing geographical information in a graph.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        device (torch.device): Device on which the module will be run.

    Attributes:
        W (nn.Linear): Linear transformation weight matrix.
    """

    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels).to(device)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weight matrices for each GCN layer.
        """
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, edge_index, dist_vec):
        """
        Forward pass of the GCN module.

        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, in_channels).
            edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
            dist_vec (torch.Tensor): Distance vector tensor of shape (num_edges,).

        Returns:
            torch.Tensor: Output tensor of shape (num_nodes, out_channels).
        """
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_weight = torch.exp(-(dist_vec ** 2))
        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)
        
        return self.W(side_embed)


class GeoGraph(nn.Module):
    """
    Graph neural network model for geographical data.

    Args:
        n_poi (int): Number of points of interest (POIs) in the graph.
        n_gcn_layers (int): Number of GCN (Graph Convolutional Network) layers.
        embed_dim (int): Dimension of the node embeddings.
        dist_edges (torch.Tensor): Tensor representing the distance edges in the graph.
        dist_vec (np.ndarray): Array representing the distance vectors in the graph.
        n_heads (int): Number of attention heads in the self-attention mechanism.
        device (torch.device): Device on which the model will be run.

    Attributes:
        n_poi (int): Number of points of interest (POIs) in the graph.
        embed_dim (int): Dimension of the node embeddings.
        n_gcn_layers (int): Number of GCN (Graph Convolutional Network) layers.
        device (torch.device): Device on which the model will be run.
        dist_edges (torch.Tensor): Tensor representing the distance edges in the graph.
        dist_vec (torch.Tensor): Tensor representing the distance vectors in the graph.
        gcn (nn.ModuleList): List of GCN modules.
        selfAttn (SelfAttn): Self-attention module.
    """

    def __init__(self, n_poi, n_gcn_layers, embed_dim, dist_edges, dist_vec, n_heads, device):
        super(GeoGraph, self).__init__()
        self.n_poi = n_poi
        self.embed_dim = embed_dim
        self.n_gcn_layers = n_gcn_layers
        self.device = device

        self.dist_edges = dist_edges.to(device)
        loop_index = torch.arange(0, n_poi).unsqueeze(0).repeat(2, 1).to(device)
        self.dist_edges = torch.cat(
            (self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1)

        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(self.n_poi)))
        self.dist_vec = torch.Tensor(dist_vec).to(device)

        # initialize GCN module, selfAttn, weights
        self.gcn = nn.ModuleList()
        for _ in range(self.n_gcn_layers):
            self.gcn.append(Geo_GCN(embed_dim, embed_dim, device).to(device))
        self.selfAttn = SelfAttn(self.embed_dim, n_heads).to(device)
        
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights in the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, data, poi_embeds):
        """
        Forward pass of the model.

        Args:
            data: Input data.
            poi_embeds: Embeddings of the points of interest (POIs).

        Returns:
            aggr_feat: Aggregated features obtained from self-attention mechanism.
            tar_embed: Embeddings of the target nodes.
        """

        enc = poi_embeds.embeds.weight
        # apply GCN layers
        for i in range(self.n_gcn_layers):
            enc = self.gcn[i](enc, self.dist_edges, self.dist_vec)
            enc = F.leaky_relu(enc)
            enc = F.normalize(enc, dim=-1)
            
        # get sequence lengths
        seq_lens = torch.bincount(data.batch)
        sections = tuple(seq_lens.cpu().numpy())
        
        # target node embeddings
        tar_embed = enc[data.poi]
        # source node embeddings
        geo_feat = enc[data.x.squeeze()]

        # apply multihead self-attention
        self_attn_feat = self.selfAttn(geo_feat, sections)
        # aggregate self-attention features to obtain semantic representation e_g,u
        aggr_feat = torch.mean(self_attn_feat, dim=1)

        return aggr_feat, tar_embed
