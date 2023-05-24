import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
import matplotlib.pyplot as plt
import networkx as nx

from GCN_Invase import InvaseGCN
from INVASE_MLP.Utilites_for_MLP import prediction_performance_metric, feature_performance_metric


def visualize(labels, g):
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=(10, 10))
    plt.axis('on')
    nx.draw_networkx(g, pos=pos, node_size=10, node_color=labels, edge_color='k',
                     arrows=False, width=1, style='dotted', with_labels=False)
    plt.show()


def to_networkx(data, node_attrs=None, edge_attrs=None):
    G = nx.DiGraph() if data.is_directed() else nx.Graph()

    G.add_nodes_from(range(data.num_nodes))
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
        if edge_attrs is not None:
            for key in edge_attrs:
                G[u][v][key] = data[key][i].item()

    if node_attrs is not None:
        for key in node_attrs:
            for i, feat in enumerate(data[key]):
                G.nodes[i][key] = feat.item()

    return G
    
    
# Model parameters
class Params:
    data_type: str
    model_type: str
    actor_h_dim: int
    critic_h_dim: int
    n_layer: int
    batch_size: int
    iteration: int
    activation: str
    learning_rate: float
    lamda: float


if __name__ == '__main__':
    # (1) Data generation
    # BA_shapes dataset to test INVASE
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=80, num_edges=30),
        motif_generator='house',
        num_motifs=300,
    )
    data = dataset[0]

    # Split data into train and test
    idx = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)

    # Visualize the graph
    #nx_G = to_networkx(data)
    #visualize(data.y, nx_G)

    # Set the parameters of the model
    params = Params()
    # Inputs for the main function
    params.data_type = 'BAShapes'
    params.model_type = 'invase'
    params.actor_h_dim = 300
    params.critic_h_dim = 400
    params.n_layer = 3
    params.batch_size = 1000
    params.iteration = 10000
    params.activation = 'relu'
    params.learning_rate = 0.001
    params.lamda = 0.1
    model_parameters = {'lamda': params.lamda,
                        'actor_h_dim': params.actor_h_dim,
                        'critic_h_dim': params.critic_h_dim,
                        'n_layer': params.n_layer,
                        'batch_size': params.batch_size,
                        'iteration': params.iteration,
                        'activation': params.activation,
                        'learning_rate': params.learning_rate,
                        'train_mask': train_idx,
                        'test_mask': test_idx}

    num_nodes = data.num_nodes
    # Dummy features for BAShapes
    data.x = torch.eye(num_nodes)

    # (2) Train INVASE or INVASE-
    model = InvaseGCN(data.x, data.edge_index, data.y, params.model_type, model_parameters)
    model.train(data.x, data.edge_index, data.y)

    # (3) Evaluate INVASE on ground truth feature importance and prediction performance
    # Compute importance score
    x_test = (data.x[test_idx])

    g_hat = model.importance_score(x_test)
    importance_score = 1. * (g_hat > 0.5)

    # Evaluate the performance of feature importance
    mean_tpr, std_tpr, mean_fdr, std_fdr = \
        feature_performance_metric(data[test_idx], importance_score)

    # Print the performance of feature importance
    print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '\%, ' + \
          'TPR std: ' + str(np.round(std_tpr, 1)) + '\%, ')
    print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '\%, ' + \
          'FDR std: ' + str(np.round(std_fdr, 1)) + '\%, ')

    # Predict labels
    y_hat = model.predict(x_test, data.edge_index)

    # Evaluate the performance of feature importance
    auc, apr, acc = prediction_performance_metric(data.y[test_idx], y_hat)

    # Print the performance of feature importance
    print('AUC: ' + str(np.round(auc, 3)) + \
          ', APR: ' + str(np.round(apr, 3)) + \
          ', ACC: ' + str(np.round(acc, 3)))

    performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
                   'mean_fdr': mean_fdr, 'std_fdr': std_fdr,
                   'auc': auc, 'apr': apr, 'acc': acc}
    print(performance)
