import numpy as np
import torch

from constants import eta, pos_num, neg_num

def sampling(adj_matrix, center_node):
    n_nodes = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1)
    probs = degrees / np.sum(degrees)
    neg_sample_probs = probs ** 0.75
    neg_sample_probs /= np.sum(neg_sample_probs)
    neg_samples = np.random.choice(n_nodes, size=(len(center_node), neg_num), replace=True, p=neg_sample_probs)

    pos_samples = []
    for i in range(len(center_node)):
        node_id = center_node[i]
        neighbor_ids = np.where(adj_matrix[node_id] == 1)[0]
        if len(neighbor_ids) > pos_num:
            neighbor_ids = np.random.choice(neighbor_ids, size=pos_num, replace=False)
        pos_samples.append(neighbor_ids)
    all_samples = np.concatenate((np.expand_dims(center_node, axis=1), np.array(pos_samples), neg_samples), axis=1)
    pos_samples = all_samples[:, 0:pos_num+1]
    neg_samples = all_samples[:, len(pos_samples)+1:]

    return pos_samples, neg_samples

def affinity(inputs1, inputs2):
    pos_score = torch.nn.functional.cosine_similarity(inputs1.unsqueeze(0), inputs2, dim=1)
    pos_score = pos_score.sum()
    pos_score = torch.log(torch.sigmoid(pos_score))
    return pos_score

def neg_affinity(inputs1, inputs2):
    neg_score = torch.nn.functional.cosine_similarity(inputs1.unsqueeze(0), inputs2, dim=1)
    neg_score = neg_score.sum()
    neg_score = inputs2.size(0) * torch.mean(torch.log(torch.sigmoid(-neg_score)))
    return neg_score

def xent_loss(adj, F):
    neg_sample_weights = 1.0
    nodes_score = []
    for i in range (len(adj)):
        id = [i]
        pos_id, neg_id = sampling(adj, id)
        pos_id = np.array(pos_id).flatten()
        neg_id = np.array(neg_id).flatten()
        input = torch.index_select(F, 0, torch.tensor(i))
        pos_sam = torch.index_select(F, 0, torch.tensor(pos_id))
        neg_sam = torch.index_select(F, 0, torch.tensor(neg_id))
        aff = affinity(input, pos_sam)
        neg_aff = neg_affinity(input, neg_sam)
        nodes_score.append(torch.mean(- aff - neg_aff))
    loss = torch.mean(torch.stack(nodes_score))
    return loss


def node_degree(node,array):
    degree =sum(array[node])
    return degree

def A(i,j,array):
    if array[i,j]==0:
        return 0
    else:
        return 1

def k(i,j,array):
    kij = node_degree(i,array) *node_degree(j,array)
    return kij

def judge_cluster(i,j,l):
    if l[i] == l[j]:
        return 1
    else:
        return 0

def modularity_loss(adj, Z):
    array = adj
    cluster = np.argmax(Z, axis=1)
    q = 0
    m = sum(sum(array)) / 2
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            if judge_cluster(i, j, cluster) != 0:
                q += (A(i, j, array) - (k(i, j, array) / (2 * m))) * judge_cluster(i, j, cluster)
    q = q / (2 * m)
    return q

def total_loss(adj, Z, F):
    x_loss = xent_loss(adj, F)
    mol_loss = modularity_loss(adj, Z)
    tot_loss = x_loss + eta * mol_loss
    return tot_loss
