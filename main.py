import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph, GraphGenerator
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import time
from GCN_Invase import InvaseGCN
from INVASE_MLP.Utilites_for_MLP import prediction_performance_metric, feature_performance_metric
from torch_geometric.data import Data
from dgl.data import TreeCycleDataset, TreeGridDataset


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
    iteration: int
    activation: str
    learning_rate: float
    lamda: float


def dgl_to_pyg(dgl_graph):
    # Get node features - assuming node feature name is 'feat'
    # Adjust the key based on your actual node feature name in the DGL graph
    node_features = dgl_graph.ndata.get('feat', None)

    # Get edge features - assuming edge feature name is 'feat'
    # Adjust the key based on your actual edge feature name in the DGL graph
    edge_features = dgl_graph.edata.get('feat', None)

    # Create a PyG Data object
    pyg_data = Data(
        x=node_features,
        edge_index=torch.stack(dgl_graph.edges()),
        edge_attr=edge_features,
        y=dgl_graph.ndata['label'],
        num_nodes=dgl_graph.number_of_nodes()
    )

    return pyg_data

if __name__ == '__main__':
    # Set a random seed, so we can always generate the same Graph
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # (1) Data generation

    # BA_shapes dataset
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=300, num_edges=5),
        motif_generator='house',
        num_motifs=80,
        transform= T.AddLaplacianEigenvectorPE(5)  # T.Constant()  => data.x = 1
    )
    data = dataset[0]

   # TreeCycle dataset
   # dataset = TreeCycleDataset()
   # data = dgl_to_pyg(dataset[0])

    # TreeGrid dataset
    #dataset = TreeGridDataset()
    #data = dgl_to_pyg(dataset[0])

    # Split data into train and test
    idx = torch.arange(data.num_nodes)

    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y, random_state=42)



    torch.manual_seed(int(time.time()))
    np.random.seed(None)

    # TODO - Use Cuda
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)

    # Visualize the graph
    # nx_G = to_networkx(data)
    #visualize(data.y, nx_G)

    # Set the parameters of the model
    params = Params()
    # Inputs for the main function
    params.data_type = 'BAShapes'
    params.model_type = 'invase'
    params.actor_h_dim = 30 #30
    params.critic_h_dim = 20# 20
    params.n_layer = 3
    params.iteration = 7000

    params.activation = 'relu'
    params.learning_rate = 0.005 # 0.005
    params.lamda = 0.001
    params.num_classes = dataset.num_classes
    model_parameters = {'lamda': params.lamda,
                        'actor_h_dim': params.actor_h_dim,
                        'critic_h_dim': params.critic_h_dim,
                        'n_layer': params.n_layer,
                        'iteration': params.iteration,
                        'activation': params.activation,
                        'learning_rate': params.learning_rate,
                        'train_mask': train_idx,
                        'test_mask': test_idx,
                        'num_classes': params.num_classes}
    num_nodes = data.num_nodes


    # Set Dummy features

    #TreeCycle és TreeGrid adathalmazokhoz
    #transform = AddLaplacianEigenvectorPE(16)
    #data = transform(data)

    #data.x = torch.eye(num_nodes)
    data.x = data.laplacian_eigenvector_pe

    print(data.x)
    # (2) Train INVASE
    model = InvaseGCN(data, params.model_type, model_parameters)
    model.train(data)

    #TODO Evaluate the model  - Ne feature importance hanem edge importance szerint kéne
    # (3) Evaluate INVASE on ground truth feature importance and prediction performance
    # Compute importance score
    g_hat = model.importance_score(data.x, data.edge_index)
    importance_score = 1. * (g_hat > 0.3)

    # Evaluate the performance of feature importance
    mean_tpr, std_tpr, mean_fdr, std_fdr = \
        feature_performance_metric(data.x.numpy(), importance_score)

    # Print the performance of feature importance
    print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '\%, ' + \
          'TPR std: ' + str(np.round(std_tpr, 1)) + '\%, ')
    print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '\%, ' + \
          'FDR std: ' + str(np.round(std_fdr, 1)) + '\%, ')

    # Set the type of the needed visualizations
    predict_params = {
        'important_node_th': 0.75,  # Az e feletti csúcsok legyenek fontosak
        'important_edge_th': 0.8,   # Az e feletti élek számítsanak fontosnak
        'edge_graph': True,        # Csak a fontos élek alapján
        'node_graph': True,        # Csak a fontos csúcsok alapján
        'k_hop_subgraph': True,     # A teszthalmaz elemeinek szomszédosságában lévő fontos és nem fontos élek
        'whole_graph': True,        # A teljes gráf, színezve rajta a fontos élek
        'unio': True,              # Fontos élek vagy fontos csúcsok
        'metszet': True}           # Fontos él és csúcs is

    # Predict labels
    y_hat = model.predict(predict_params, data)

    # Evaluate the performance of feature importance
    auc, apr, acc = prediction_performance_metric(data.y[test_idx], y_hat[test_idx])

    # Print the performance of feature importance
    print('AUC: ' + str(np.round(auc, 3)) + \
          ', APR: ' + str(np.round(apr, 3)) + \
          ', ACC: ' + str(np.round(acc, 3)))

    performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
                   'mean_fdr': mean_fdr, 'std_fdr': std_fdr,
                   'auc': auc, 'apr': apr, 'acc': acc}
    print(performance)
