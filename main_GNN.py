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
from GNN_Invase import InvaseGNN
from Utilities import prediction_performance_metric, feature_performance_metric
from torch_geometric.data import Data
from dgl.data import TreeCycleDataset, TreeGridDataset
from torch_geometric.utils import subgraph


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


def edge_mask_from_node_mask(edge_index, node_mask):
    indices = [i for i, x in enumerate(node_mask) if x == 1]
    sub_edge_index, _ = subgraph(indices, edge_index)

    # Convert edge indices to a set of tuples for easy comparison
    S_edges = set(tuple(x) for x in sub_edge_index.t().tolist())
    # Initialize mask with zeros
    mask = torch.zeros(edge_index.size(1), dtype=torch.int32)

    # Fill in the mask
    for i, edge in enumerate(edge_index.t().tolist()):
        if tuple(edge) in S_edges:
            mask[i] = 1

    return sub_edge_index, mask


def load_dataset(dataset_name):
    if dataset_name == "ba-shapes":
        # BA_shapes dataset from PyG
        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=300, num_edges=5),
            motif_generator='house',
            num_motifs=80,
            transform=T.AddLaplacianEigenvectorPE(1, is_undirected=True)  # T.Constant()  => data.x = 1
        )
        data = dataset[0]
        data.x = data.laplacian_eigenvector_pe
        data.num_classes = dataset.num_classes
        return data

    if dataset_name == "tree-cycle":
        dataset = TreeCycleDataset(force_reload=True, seed=17)
        data = dgl_to_pyg(dataset[0])
        a, b = data.edge_index

        # Set Dummy features
        # transform = AddLaplacianEigenvectorPE(1)
        # data = transform(data)

        reverse_edges = torch.stack([b, a], dim=0)
        data.edge_index = torch.cat((data.edge_index, reverse_edges), dim=1).to(data.edge_index.dtype)
        data.num_classes = dataset.num_classes
        return data

    if dataset_name == "tree-grid":
        # TreeGrid dataset
        dataset = TreeGridDataset(force_reload=True, seed=17)
        data = dgl_to_pyg(dataset[0])
        a, b = data.edge_index

        # Set Dummy features
        # transform = AddLaplacianEigenvectorPE(5)
        # data = transform(data)

        reverse_edges = torch.stack([b, a], dim=0)
        data.edge_index = torch.cat((data.edge_index, reverse_edges), dim=1).to(data.edge_index.dtype)
        data.num_classes = dataset.num_classes
        return data

    print("Dataset not found")


if __name__ == '__main__':
    # Set a random seed, so we can always generate the same Graph
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    np.random.seed(17)

    # Set the parameters of the model
    params = Params()
    params.data_type = "ba-shapes" #tree-cycle, tree-grid
    # (1) Load dataset
    data = load_dataset(params.data_type)

    # Create a new variable for edges, without uplicate edges
    edges = data.edge_index.numpy().T
    sorted_edges = np.sort(edges, axis=1)
    _, unique_indices = np.unique(sorted_edges, axis=0, return_index=True)
    unique_mask = np.zeros(edges.shape[0], dtype=bool)
    unique_mask[unique_indices] = True
    data.unique_edge_index = data.edge_index[:, unique_mask]

    a, b = data.unique_edge_index

    reverse_edges = torch.stack([b, a], dim=0)
    #Change the order of the edge index, to match it with the unique edge index
    data.edge_index = torch.cat((data.unique_edge_index, reverse_edges), dim=1).to(data.edge_index.dtype)
    torch.set_printoptions(threshold=15000)

    # Split data into train and test
    idx = torch.arange(data.num_nodes)

    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y, random_state=42)

    torch.manual_seed(int(time.time()))
    np.random.seed(None)

    # Visualize the graph
    # nx_G = to_networkx(data)
    # visualize(data.y, nx_G)

    # Set the parameters of the model
    params = Params()
    # Inputs for the main function
    params.model_type = 'invase'
    params.critic_model = 'GIN'  # 'GCN'
    params.critic_h_dim = 10
    params.critic_n_layer = 2
    params.learning_rate = 0.0008
    params.iteration = 8000
    params.activation = 'relu'
    params.actor_model= 'GIN' # 'GCN'
    params.actor_h_dim = 10
    params.actor_n_layer = 2
    params.lamda = 0.08  # 0.005000
    params.num_classes = data.num_classes
    model_parameters = {'lamda': params.lamda,
                        'critic_h_dim': params.critic_h_dim,
                        'actor_h_dim': params.actor_h_dim,
                        'critic_n_layer': params.critic_n_layer,
                        'actor_n_layer': params.actor_n_layer,
                        'iteration': params.iteration,
                        'activation': params.activation,
                        'learning_rate': params.learning_rate,
                        'actor_model': params.actor_model,
                        'critic_model': params.critic_model,
                        'train_mask': train_idx,
                        'test_mask': test_idx,
                        'num_classes': params.num_classes}
    num_nodes = data.num_nodes

    # Set the ground truth explainability importance scores
    data.gt_importance_x = [0 if y == 0 else 1 for y in data.y]
    # Transform ground Truth importance scores from node level to edge level
    data.gt_importance_edges, data.gt_importance_edges_mask = edge_mask_from_node_mask(data.unique_edge_index,
                                                                                       data.gt_importance_x)
    # (2) Train INVASE
    model = InvaseGNN(data, params.model_type, model_parameters)
    model.train(data)

    # (3) Evaluate INVASE on ground truth edge importance and prediction performance
    # Compute importance score
    important_node_th = 0.75  # Az e feletti csúcsok legyenek fontosak
    important_edge_th = 0.5  # Az e feletti élek számítsanak fontosnak
    importance_mask = model.importance_score(data, important_edge_th)
    # # Evaluate the performance of edge importance  (ground truth, importance score from the actor)
    mean_tpr, std_tpr, mean_fdr, std_fdr = \
        feature_performance_metric(data.gt_importance_edges_mask, importance_mask)

    # Print the performance of edge importance
    print('TPR mean: ' + str(np.round(mean_tpr, 1)) + '\%, ' + \
          'TPR std: ' + str(np.round(std_tpr, 1)) + '\%, ')
    print('FDR mean: ' + str(np.round(mean_fdr, 1)) + '\%, ' + \
          'FDR std: ' + str(np.round(std_fdr, 1)) + '\%, ')

    # Predict labels
    baseline_pred, y_hat, y_hat_inverse, edge_selection, edge_selection_probability = model.predict(data,
                                                                                                    important_node_th,
                                                                                                    important_edge_th,
                                                                                                    need_randoms=True)
    # Visualize Importance predictions
    # Set the type of the needed visualizations
    visualize_params = {
        'edge_graph': True,  # Csak a fontos élek alapján
        'node_graph': False,  # Csak a fontos csúcsok alapján
        'k_hop_subgraph': False,  # A teszthalmaz elemeinek szomszédosságában lévő fontos és nem fontos élek
        'whole_graph': True,  # A teljes gráf, színezve rajta a fontos élek
        'unio': False,  # Fontos élek vagy fontos csúcsok
        'metszet': False}  # Fontos él és csúcs is

    model.show_graphs(data, edge_selection, torch.ones(data.x.shape), visualize_params, edge_selection_probability)

    # EVALUATE

    # Sparsity
    Sparsity = 1 - (edge_selection.sum() / data.edge_index.size(1))
    print("Sparsity: ", Sparsity.item())
    # Fidelity score
    # Acc of the original labels
    correct_baseline = int((baseline_pred[test_idx] == data.y[test_idx]).sum())
    baseline_acc = correct_baseline / test_idx.size(0)
    # Acc of the graph only with the important edges
    correct_y_hat = int((y_hat[test_idx] == data.y[test_idx]).sum())
    acc_important_edges = correct_y_hat / test_idx.size(0)
    # Acc of the graph without the important edges
    correct_y_hat_inverse = int((y_hat_inverse[test_idx] == data.y[test_idx]).sum())
    acc_unimportant_edges = correct_y_hat_inverse / test_idx.size(0)

    print("ACC of the whole graph: ", baseline_acc, ",\n ACC with only the important edges", acc_important_edges,
          "\n ACC with only the unimportant edges", acc_unimportant_edges)

    # Fidelity score
    Fidelity_plus = baseline_acc - acc_unimportant_edges
    Fidelity_minus = baseline_acc - acc_important_edges
    print("Fidelity plus score: ", Fidelity_plus, "\nFidelity minus score: ", Fidelity_minus)

    # # # Evaluate the performance of feature importance
    # auc, apr, acc = prediction_performance_metric(data.y[test_idx], y_hat[test_idx])
    #
    # # Print the performance of feature importance
    # print('AUC: ' + str(np.round(auc, 3)) + \
    #        ', APR: ' + str(np.round(apr, 3)) + \
    #        ', ACC: ' + str(np.round(acc, 3)))

