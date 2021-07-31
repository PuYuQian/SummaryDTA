import sys, os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from gnn import GNNNet
from utils import *
from emetrics import *
from data_process import create_dataset_for_5folds


datasets = [['davis', 'kiba'][int(sys.argv[1])]]
cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]
print('cuda_name:', cuda_name)
fold = [0, 1, 2, 3, 4][1]
file_name = f'{sys.argv[3]}'

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
log_dir = 'runs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir, filename_suffix=file_name)
# Main program: iterate over different datasets

USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet(device)
model.to(device)
model_st = GNNNet.__name__
model.load_state_dict(torch.load('models/model_GNNNet_davis.model', map_location=cuda_name), strict=False)
# model.load_state_dict(torch.load('models/model_GNNNet_kiba.model', map_location=torch.device('cpu')), strict=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=1)
for dataset in datasets:
    train_data, valid_data = create_dataset_for_5folds(dataset, fold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    model_file_name = 'models/'+file_name+'_model_' + model_st + '_' + dataset + '.model'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, writer, TRAIN_BATCH_SIZE)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val = get_mse(G, P)
        writer.add_scalar('Valid/Loss', val, epoch)
        print('valid result:', val, best_mse)
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset)
        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset)
        scheduler.step(val)
