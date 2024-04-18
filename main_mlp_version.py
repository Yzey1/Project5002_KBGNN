import math
import os
import torch
import random
import argparse
import logging
import pickle
from dataset import MyDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from GeoGraph import GeoGraph
from SeqGraph import SeqGraph
from consistency import consistencyLoss
import torch.nn as nn

ARG = argparse.ArgumentParser()
ARG.add_argument('--epoch', type=int, default=140,
                 help='Max epoch num.')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed.')
ARG.add_argument('--batch', type=int, default=128,
                 help='Training batch size.')
ARG.add_argument('--data', type=str, default='nyc',
                 help='Training dataset.')
ARG.add_argument('--gpu', type=int, default=None,
                 help='Denote training device. GPU is denoted by the index (e.g., 0, 1).')
ARG.add_argument('--patience', type=int, default=10,
                 help='Early stopping patience.')
ARG.add_argument('--embed', type=int, default=64,
                 help='Embedding dimension.')
ARG.add_argument('--gcn_num', type=int, default=2,
                 help='Num of GCN.')
ARG.add_argument('--max_step', type=int, default=2,
                 help='Steps of random walk.')
ARG.add_argument('--hid_graph_num', type=int, default=16,
                 help='Num of hidden graphs.')
ARG.add_argument('--hid_graph_size', type=int, default=10,
                 help='Size of hidden graphs')
ARG.add_argument('--weight_decay', type=float, default=5e-4,
                 help='Weight decay rate')
ARG.add_argument('--lr', type=float, default=1e-3,
                 help='Learning rate.')
ARG.add_argument('--log', type=str, default=None,
                 help='Log file path.')
ARG.add_argument('--con_weight', type=float, default=0.01,
                 help='Weight of consistency loss')
ARG.add_argument('--compress_memory_size', type=int, default=12800,
                 help='Memory bank size')
ARG.add_argument('--compress_t', type=float, default=0.01,
                 help='Softmax temperature')
ARG.add_argument('--train_percentage', type=float, default=0.1,
                 help='Percentage of training set')

ARG = ARG.parse_args()


def eval_model(Seq_encoder, Geo_encoder, Poi_embeds, MLP, dataset, arg, device):
    loader = DataLoader(dataset, arg.batch, shuffle=True)
    preds, labels, pois = [], [], []

    Seq_encoder.eval()
    Geo_encoder.eval()
    MLP.eval()

    with torch.no_grad():
        for batch in loader:
            e_s, _ = Seq_encoder(batch.to(device), Poi_embeds)
            e_g, _, h_t = Geo_encoder(batch.to(device), Poi_embeds)
            logit = MLP(e_g, e_s, h_t)
            logit = torch.sigmoid(logit).squeeze(
            ).clone().detach().cpu().numpy()

            preds.append(logit)
            labels.append(batch.y.squeeze().cpu().numpy())

    preds = np.concatenate(preds, 0, dtype=np.float64)
    labels = np.concatenate(labels, 0, dtype=np.float64)

    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds)

    return auc, logloss


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class EmbeddingLayer(nn.Module):
    '''Embedding layer for POI. 

    Args:
        n_poi (int): Number of POI.
        embed_dim (int): Embedding dimension.

    Output:
        torch.Tensor: Embedding vector of POI.
    '''

    def __init__(self, n_poi, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeds = nn.Embedding(n_poi, embed_dim)
        nn.init.xavier_normal_(self.embeds.weight)

    def forward(self, idx):
        return self.embeds(idx)


class integrative_pred(nn.Module):
    def __init__(self, embed_dim):
        super(integrative_pred, self).__init__()
        self.embed_dim = embed_dim

        self.predictor = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, e_g, e_s, h_t):
        flat_input = torch.cat((e_g, e_s, h_t), dim=-1)
        pred_logits = self.predictor(flat_input)

        return pred_logits


def train_test(tr_set, va_set, te_set, arg, dist_edges, dist_vec, device):
    Seq_encoder = SeqGraph(n_user, n_poi, arg.max_step, arg.embed,
                           arg.hid_graph_num, arg.hid_graph_size, device).to(device)
    Geo_encoder = GeoGraph(n_user, n_poi, arg.gcn_num,
                           arg.embed, dist_edges, dist_vec, device).to(device)
    Poi_embeds = EmbeddingLayer(n_poi, arg.embed).to(device)
    Sim_criterion = consistencyLoss(
        arg.embed, arg.compress_memory_size, arg.compress_t, device).to(device)

    # MLP
    MLP = integrative_pred(arg.embed).to(device)

    opt = torch.optim.Adam([
        {'params': Seq_encoder.parameters()},
        {'params': Geo_encoder.parameters()},
        {'params': Poi_embeds.parameters()},
        {'params': MLP.parameters()}], lr=arg.lr)  # , weight_decay=arg.weight_decay)

    batch_num = math.ceil(len(tr_set) / arg.batch)
    train_loader = DataLoader(tr_set, arg.batch, shuffle=True)
    bank_loader = DataLoader(tr_set, arg.batch, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch = 0., 0
    test_auc, test_loss = 0., 0

    for epoch in range(arg.epoch):
        Seq_encoder.train()
        Geo_encoder.train()
        MLP.train()
        for bn, (trn_batch, bnk_batch) in enumerate(zip(train_loader, bank_loader)):
            trn_batch, bnk_batch = trn_batch.to(device), bnk_batch.to(device)
            label = trn_batch.y.float()

            seq_trn_enc, _ = Seq_encoder(trn_batch, Poi_embeds)
            seq_bnk_enc, _ = Seq_encoder(bnk_batch, Poi_embeds)

            geo_trn_enc, _, geo_tar = Geo_encoder(trn_batch, Poi_embeds)
            geo_bnk_enc, _, _ = Geo_encoder(bnk_batch, Poi_embeds)

            # MLP
            final_pred = MLP(geo_trn_enc, seq_trn_enc, geo_tar)
            loss_rec = criterion(final_pred.squeeze(), label)

            unsup_loss = Sim_criterion(seq_bnk_enc, geo_bnk_enc)
            # loss = seq_sup_loss + geo_sup_loss + arg.con_weight * unsup_loss
            loss = loss_rec + arg.con_weight * unsup_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (bn + 1) % 20 == 0:
                logging.info(
                    f'Epoch: {epoch + 1} / {arg.epoch} Batch: {bn + 1} / {batch_num}, loss: {loss.item()} = Rec: {loss_rec.item()} + {ARG.con_weight} * Con: {unsup_loss.item()}')

        # validation
        auc, logloss = eval_model(
            Seq_encoder, Geo_encoder, Poi_embeds, MLP, va_set, arg, device)
        logging.info('')
        logging.info(
            f'''Epoch: {epoch + 1} / {arg.epoch}, validation AUC: {auc}, validation logloss: {logloss}''')

        # update best epoch
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            test_auc, test_loss = eval_model(
                Seq_encoder, Geo_encoder, Poi_embeds, MLP, te_set, arg, device)

        # early stopping
        if epoch - best_epoch == arg.patience:
            logging.info(
                f'Stop training after {arg.patience} epochs without improvement.')
            break

        logging.info(
            f'Best validation AUC: {best_auc} at epoch {best_epoch + 1}\n')

    logging.info(f'Training finished, best epoch {best_epoch + 1}')
    logging.info(
        f'Validation AUC: {best_auc}, Test AUC: {test_auc}, Test logloss: {test_loss}')


if __name__ == '__main__':
    set_seed(ARG.seed)

    LOG_FORMAT = "%(asctime)s  %(message)s"
    DATE_FORMAT = "%m/%d %H:%M:%S"
    if ARG.log is not None:
        logging.basicConfig(filename=ARG.log, level=logging.DEBUG,
                            format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format=LOG_FORMAT, datefmt=DATE_FORMAT)

    with open(f'./processed_data/{ARG.data}/raw/val.pkl', 'rb') as f:
        tmp = pickle.load(f)
        n_user, n_poi = pickle.load(f)
        del tmp

    train_set = MyDataset(f'./processed_data/{ARG.data}', set='train')
    train_set = train_set[:int(len(train_set) * ARG.train_percentage)]
    test_set = MyDataset(f'./processed_data/{ARG.data}', set='test')
    val_set = MyDataset(f'./processed_data/{ARG.data}', set='val')

    with open(f'./processed_data/{ARG.data}/raw/dist_graph.pkl', 'rb') as f:
        dist_edges = torch.LongTensor(pickle.load(f))
        dist_nei = pickle.load(f)
    dist_vec = np.load(f'./processed_data/{ARG.data}/raw/dist_on_graph.npy')

    logging.info(f'Data loaded.')
    logging.info(f'user: {n_user}\tpoi: {n_poi}')
    device = torch.device(
        'cpu') if ARG.gpu is None or not torch.cuda.is_available() else torch.device(f'cuda:{ARG.gpu}')
    logging.info(f'Device: {device}')
    train_test(train_set, val_set, test_set, ARG, dist_edges, dist_vec, device)
