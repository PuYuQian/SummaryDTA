import torch
import torch.nn as nn
import numpy as np
from resnet18 import resnet18
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, GlobalAttention as ga,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj


# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, device, n_output=1, hidden_size=128, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.prot_rnn = nn.RNN(1280, hidden_size, 1)
        self.resnet = resnet18()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc = nn.Sequential(nn.Linear(1410, 512),nn.ReLU(),nn.Dropout(dropout),
                                nn.Linear(512, self.n_output))

        self.encoder1 = nn.Sequential(nn.Linear(257 * 129, 1024), nn.ReLU(), nn.Dropout(dropout))
        # self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, data_mol, data_pro, data_pro_len, data_pro_cm):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch

        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len,descending=True)[::-1][1], torch.argsort(
            -data_pro_len)
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        # print(pro_seq_lengths)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt = self.prot_rnn(xt)[0]
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=1024)[0]
        xt = xt.index_select(0, pro_idx_unsort)

        xt = xt.mean(1)
        # print(xt.shape)
        xt_cm = self.resnet(data_pro_cm)

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # concat
        xc = torch.cat((xt, xt_cm), 1)
        o1 = torch.cat((xc, torch.FloatTensor(xc.shape[0], 1).fill_(1).cuda()), 1)  # o1: 5, 257
        # print("o1:",o1.shape)
        o2 = torch.cat((x, torch.FloatTensor(x.shape[0], 1).fill_(1).cuda()), 1)  # o2 :5, 129
        # print("o2:",o2.shape)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(
            start_dim=1)  # o1.unsqueeze(2):(1,513,1) o2.unsqueeze(1):(1,1,513) o12:5, 33153
        # print("o12:", o12.shape)
        out = self.dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2), 1)
        # print("out:", out.shape) # 5, 1410
        # add some dense layers
        out = self.fc(out)
        return out
