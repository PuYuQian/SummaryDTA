# How to use SummaryDTA for your research
Summary of the second academic year: drug-target binding affinity prediction

***The whole content is actually an improvement on DGraphDTA, but unfortunately, the ideas are not better to improve the accuracy, so the work of this stage is summarized as follows, hope to inspire you:***

The target sequences obtain prior representation on the pre-trained protein sequence model,which called ESM, and the sequence feature vectors are obtained through the variable-length RNN network and averaging operation. The targets also use the concat maps as the structure information, which describe the interaction information between the residues. The three residual blocks obtain the structure feature vector, and finally represent the target information through the concat operation. The drug SMILESs are constructed graphs through the node feature and adjacency matrix, which are obtained by the RDKit tool. Hence, the three GCN layers and the max-pooling operation are designed to obtain the graph representation vector of the drugs. Finally, the Kronecker Product is used to perform feature fusion on the candidate drug-target pair, and the final prediction binding affinity values are output through the linear layers.

![](https://github.com/PuYuQian/SummaryDTA/fig/network.png)
> The flowchart of our proposed method.

##Usage
###INSTALL
 I am using Ubuntu Linux 18.04 LTS with an NVIDIA Tesla V100 GPU and 32GB RAM.
###Required
- [Python] (https://www.python.org) (3.7.10). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.anaconda.com/download/) (4.9.2).
- [PyTorch] (https://pytorch.org/) (1.7.1).
- [PyTorch Geometric (PyG)] (https://github.com/rusty1s/pytorch_geometric).


    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html
	pip install torch-geometric

- [RDKit] (https://anaconda.org/rdkit/rdkit) (2020.09.1.0).
- [Tensorboard] (`pip install tensorboard`) (2.5.0)
###Run
---
The *data_toy* folder contains samples of the KIBA dataset. By using it, you can see the results more quickly.
1. Train
`python training_5folds.py arg1 arg2 arg3`
arg1: choose dataset, davis = 0 and kiba = 1
arg2: choose GPU, 0 1 2 3
arg3: the name of output model , e.g. 'test'
e.g. `python training_5folds.py 1 0 test`
2. Test
`python test.py arg1 arg2 arg3`
arg1: choose dataset, davis = 0 and kiba = 1
arg2: choose GPU, 0 1 2 3
arg3: the name of model to load, e.g. 'test'
e.g. `python test.py 1 0 test`

The output results are saved in the *results* folder, and the output loss curve is stored in the *runs* folder.
For your convenience, I have uploaded the trained model and data to the following links:
[Outside China] (https://1drv.ms/u/s!Ah5yIdya3kMUi3iFo1GuY3iEtTpU?e=5b5cyL)
[China] (https://www.aliyundrive.com/s/7k12eDy52NX)



