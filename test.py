import os
import sys
import torch
import numpy as np
from random import shuffle
from torch_geometric.data import Batch

from emetrics import get_aupr, get_cindex, get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
from utils import *
from scipy import stats
from gnn import GNNNet
from data_process import create_dataset_for_test


def predicting(model, device, loader, file_name):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for index, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            data_pro_cm = data[3].to(device)
            output = model(data_mol, data_pro, data_pro_len, data_pro_cm)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)

    np.save(f'results/pred_{file_name}.npy', total_preds.numpy())
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def load_model(model_path):
    model = torch.load(model_path)
    return model

def calculate_metrics(Y, P, dataset='davis', result_file_name=None):
    # aupr = get_aupr(Y, P)
    cindex = get_cindex(Y, P)  # DeepDTA
    cindex2 = get_ci(Y, P)  # GraphDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)
    print('cindex2', cindex2)
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)

    result_str = ''
    result_str += dataset + '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'w').writelines(result_str)


if __name__ == '__main__':
    dataset = ['davis', 'kiba'][int(sys.argv[1])]  # dataset selection
    model_st = GNNNet.__name__
    print('dataset:', dataset)

    cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]  # gpu selection
    print('cuda_name:', cuda_name)
    file_name = f'{sys.argv[3]}'

    TEST_BATCH_SIZE = 512
    models_dir = 'models'
    results_dir = 'results'

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    model_file_name = 'models/'+file_name+'_model_' + model_st + '_' + dataset + '.model'
    result_file_name = 'results/' + file_name + '_result_' + model_st + '_' + dataset + '.txt'

    model = GNNNet(device)
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    # model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')))
    test_data = create_dataset_for_test(dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)

    Y, P = predicting(model, device, test_loader, file_name)
    calculate_metrics(Y, P, dataset, result_file_name)
    # plot_density(Y, P, fold, dataset)
