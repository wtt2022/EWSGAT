#Title: Weighted Graph Structure Learning with Denoising Sparse Attention for Graph Clustering

#Requirements
dgl 1.0.1
h5py 3.1.0
igraph 0.10.4
keras 2.10.0
networkx 2.6.3
numpy 1.22.4
scipy 0.16.0
tensorflow 2.10.0
torch 1.12.1

#Dataset(Movie)
movie_gra.csv: <movie id1, movie id2, weight> 
movie_info.csv: <movie id, label> 

#Code
constant.py: Setting of hyperparameters, including learning rate, number of hidden layers, etc.
entmax.py: Implementation of the activation function a-entmax.
FCM.py: Implementing fuzzy C-means clustering.
layers.py: Implementing a multi-headed attention mechanism.
losses.py: Implementing the loss function.
model.py: Multi-layer GAT learning node representation and community detection results.
metrics.py: Achievement of evaluation indicators ACC and F1.
trainer.py: Implementing the training of the model.
utils.py: Reading data to enable the construction of social networks.
run.py: Main function

