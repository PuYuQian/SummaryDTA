import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch

from utils import *


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index


# to judge whether the required files exist
def valid_target(key, dataset):
    esm_dir = 'data/' + dataset + '/esm'
    esm_file = os.path.join(esm_dir, key + '.pt')
    contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, key + '.pth')
    if os.path.exists(contact_file or esm_file):
        return True
    else:
        return False

def target_matrics(key, dataset):
    esm_path = 'data/' + dataset + '/esm'
    esm_file = os.path.join(esm_path, key + '.pt')
    input_feature = torch.load(esm_file)

    contac_path = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contac_path, key + '.pth')
    contact_map = torch.load(contact_file)
    contact_map += torch.eye(contact_map.shape[0])
    return input_feature['feature'], input_feature['size'], contact_map

def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def create_dataset_for_test(dataset):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    test_fold = json.load(open(dataset_path + 'folds/test_fold_setting0.txt'))
    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    # load contact and aln
    esm_path = 'data/' + dataset + '/esm'
    esm_list = []
    for key in proteins:
        esm_list.append(os.path.join(esm_path, key + '.pt'))

    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_test_count = 0
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    temp_test_entries = []
    for pair_ind in range(len(rows)):
        # if the required files is not exist, then pass
        if not valid_target(prot_keys[cols[pair_ind]], dataset):
            continue
        ls = []
        ls += [drugs[rows[pair_ind]]]
        ls += [prots[cols[pair_ind]]]
        ls += [prot_keys[cols[pair_ind]]]
        ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        temp_test_entries.append(ls)
        valid_test_count += 1
    csv_file = 'data/' + dataset + '_test.csv'
    data_to_csv(csv_file, temp_test_entries)
    print('dataset:', dataset)
    print('test entries:', len(test_fold), 'effective test entries', valid_test_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        target_graph[key] = target_matrics(key, dataset)

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_test.csv' or data/kiba_test.csv'
    df_test = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prot_keys, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_key']), list(
        df_test['affinity'])
    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_test', xd=test_drugs, y=test_Y,
                              target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph)

    return test_dataset


def create_dataset_for_5folds(dataset, fold=0):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(dataset_path + 'folds/train_fold_setting0.txt'))
    train_fold_origin = [e for e in train_fold_origin]  # for 5 folds

    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    # load contact and aln
    esm_path = 'data/' + dataset + '/esm'
    esm_list = []
    for key in proteins:
        esm_list.append(os.path.join(esm_path, key + '.pt'))

    # load train,valid and test entries
    train_folds = []
    valid_fold = train_fold_origin[fold]  # one fold
    for i in range(len(train_fold_origin)):  # other folds
        if i != fold:
            train_folds += train_fold_origin[i]

    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'valid']
    valid_train_count = 0
    valid_valid_count = 0
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_folds], cols[train_folds]
            train_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # ensure the contact and aln files exists
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_fold_entries.append(ls)
                valid_train_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, train_fold_entries)
        elif opt == 'valid':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[valid_fold], cols[valid_fold]
            valid_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                valid_fold_entries.append(ls)
                valid_valid_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, valid_fold_entries)
    print('dataset:', dataset)
    # print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))

    # entries with protein contact and aln files are marked as effiective
    print('fold:', fold)
    print('train entries:', len(train_folds), 'effective train entries', valid_train_count)
    print('valid entries:', len(valid_fold), 'effective valid entries', valid_valid_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        target_graph[key] = target_matrics(key, dataset)

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    train_csv = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'train' + '.csv'
    df_train_fold = pd.read_csv(train_csv)
    train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y, smile_graph=smile_graph, target_graph=target_graph)


    df_valid_fold = pd.read_csv('data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'valid' + '.csv')
    valid_drugs, valid_prots_keys, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity'])
    valid_drugs, valid_prots_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots_keys), np.asarray(
        valid_Y)
    valid_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=valid_drugs,
                               target_key=valid_prots_keys, y=valid_Y, smile_graph=smile_graph,
                               target_graph=target_graph)
    return train_dataset, valid_dataset
