# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import networkx as nx
from torch_geometric.utils import k_hop_subgraph

from Utilities import *
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCN, GIN, MLP
# from main import to_networkx, visualize
import matplotlib.pyplot as plt

def visualize(labels, g, title="Graph", edge='black'):
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('on')
    nx.draw_networkx(g, pos=pos, node_size=10, node_color=labels, edge_color=edge,  # edge_cmap = plt.cm.Blues,
                     arrows=False, width=1, style='solid', with_labels=False)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1)))
    plt.savefig(title + ".png")


def to_networkx(data, node_attrs=None, edge_attrs=None):
    G = nx.Graph()  # nx.DiGraph() if data.is_directed() else

    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    # for i, (u, v) in enumerate(data.edge_index.t().tolist()):
    #
    #     G.add_edge(u, v)
    #     if edge_attrs is not None:
    #         for key in edge_attrs:
    #             G[u][v][key] = data[key][i].item()
    #
    # if node_attrs is not None:
    #     for key in node_attrs:
    #         for i, feat in enumerate(data[key]):
    #             G.nodes[i][key] = feat.item()

    return G


"""
**INVASE algorithm implementation in Pytorch for GCN**
Based on
Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "INVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
"""


# Costume loss function
class CrossEntropy:
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self, y_ground_truth, out):
        result = -torch.sum(y_ground_truth * torch.log(out + 1e-8), dim=1)
        if self.reduction == 'mean':
            result = torch.mean(result)
        return result


# **INVASE Model**
class InvaseGCN:
    def __init__(self, data, model_type, model_parameters):
        x_train = data.x
        A_train = data.edge_index
        y_train = data.y
        self.lamda = model_parameters['lamda']
        self.critic_h_dim = model_parameters['critic_h_dim']
        self.n_layer = model_parameters['n_layer']
        self.iteration = model_parameters['iteration']
        if model_parameters['activation'] == "relu":
            self.activation = nn.ReLU()
        self.learning_rate = model_parameters['learning_rate']
        self.num_nodes = data.num_nodes
        self.dim = data.num_features
        self.label_dim = model_parameters['num_classes']
        self.model_type = model_type
        self.train_mask = model_parameters['train_mask']
        self.test_mask = model_parameters['test_mask']

        # Build and compile critic
        self.critic = self.build_critic()
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        # self.critic.loss = CrossEntropy(reduction='mean')

        # Build and compile the actor
        self.actor = self.build_actor(self)
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.actor.loss_invase = self.actor_loss

        if self.model_type == 'invase':
            # Build and compile the baseline
            self.baseline = self.build_baseline()
            self.baseline.optimizer = torch.optim.Adam(self.baseline.parameters(), self.learning_rate)
            # self.baseline.loss = CrossEntropy(reduction='mean')

    def actor_loss(self, y_final, y, edge_selection, edge_selection_probability):

        x = y_final[:, :self.dim]
        # Critic output
        critic_out = y_final[:, self.dim:(self.dim + self.label_dim)]
        if self.model_type == 'invase':
            # Baseline output
            baseline_out = y_final[:, (self.dim + self.label_dim):(self.dim + 2 * self.label_dim)]

        y_ground_truth = y
        # Critic loss
        critic_loss = F.cross_entropy(self.critic.output[self.train_mask], y_ground_truth[self.train_mask])

        if self.model_type == 'invase':
            # Baseline loss
            baseline_loss = F.cross_entropy(self.baseline.output[self.train_mask], y_ground_truth[self.train_mask])
            # Reward
            Reward = -(critic_loss - baseline_loss.detach())

        else:
            Reward = -critic_loss

        loss1 = Reward * \
               torch.mean(edge_selection * torch.log(edge_selection_probability + 1e-8) + \
               (1 - edge_selection) * torch.log(1 - edge_selection_probability + 1e-8))
        loss2 = self.lamda * torch.mean(edge_selection_probability)

       # print(loss1, loss2)

        loss = loss1 - loss2

        custom_actor_loss = -loss

        return custom_actor_loss +1

    def build_actor(self, params):

        # GCN model
        # class ActorGCN(nn.Module):
        #     def __init__(self, params):
        #         super(ActorGCN, self).__init__()
        #         # Params
        #         self.activation = params.activation
        #         self.n_layer = params.n_layer
        #         self.label_dim = params.label_dim
        #         self.dim = params.dim
        #         self.baseline_h_dim = params.critic_h_dim  # same as the critic
        #
        #         # Layers
        #         self.GCNConv_in = GCNConv(self.dim, self.baseline_h_dim)
        #         self.GCNConv_hidden = GCNConv(self.baseline_h_dim, self.baseline_h_dim)
        #         self.GCNConv_out = GCNConv(self.baseline_h_dim, self.label_dim)
        #
        #
        #     def forward(self, feature, edge_index):
        #         x = self.activation(self.GCNConv_in(feature, edge_index))
        #         for i in range(self.n_layer - 2):
        #             x = self.activation(self.GCNConv_hidden(x, edge_index))
        #         y_hat = (self.GCNConv_out(x, edge_index))
        #         return torch.sigmoid(torch.matmul(y_hat, y_hat.t()))
        #
        # actor_model = ActorGCN(self)
        # return actor_model

        # # Autoencoder model
        encoder = GIN(in_channels=self.dim, hidden_channels=10, num_layers=2) #GCN(self.dim, 20, 3)
        actor_model = GAE(encoder)
        return actor_model

    def build_critic(self):

        class CriticGCN(nn.Module):
            def __init__(self, params):
                super(CriticGCN, self).__init__()
                # Params
                self.activation = params.activation
                self.n_layer = params.n_layer
                self.label_dim = params.label_dim
                self.dim1 = params.dim
                self.critic_h_dim = params.critic_h_dim

                # Layers
                self.GCNConv_in = GINConv(MLP([self.dim1, self.critic_h_dim, self.critic_h_dim])) #GCNConv(self.dim1, self.critic_h_dim)
                self.GCNConv_hidden = GINConv(MLP([self.critic_h_dim, self.critic_h_dim, self.critic_h_dim])) #GCNConv(self.critic_h_dim, self.critic_h_dim)
                self.GCN_Conv_out = GINConv(MLP([self.critic_h_dim, self.critic_h_dim, self.label_dim])) #GCNConv(self.critic_h_dim, self.label_dim)

            def forward(self, feature, edge_index, edge_selection):
                # Element wise multiplication
                # Use only the selected features and edges

                selected_edge_index = edge_index[:, edge_selection.to(torch.bool)]
               #
                # critic_model_A_input = (edge_index * A_selection).float()
                x = self.activation(self.GCNConv_in(feature, selected_edge_index))
                for i in range(self.n_layer - 2):
                    x = self.activation(self.GCNConv_hidden(x, selected_edge_index))
                y_hat = nn.Softmax(dim=1)(self.GCN_Conv_out(x, selected_edge_index))
                return y_hat

        critic_model = CriticGCN(self)
        return critic_model

    def build_baseline(self):

        class BaselineGCN(nn.Module):
            def __init__(self, params):
                super(BaselineGCN, self).__init__()
                # Params
                self.activation = params.activation
                self.n_layer = params.n_layer
                self.label_dim = params.label_dim
                self.dim = params.dim
                self.baseline_h_dim = params.critic_h_dim  # same as the critic

                # Layers
                self.GCNConv_in = GINConv(MLP([self.dim, self.baseline_h_dim, self.baseline_h_dim])) #GCNConv(self.dim, self.baseline_h_dim)
                self.GCNConv_hidden = GINConv(MLP([self.baseline_h_dim, self.baseline_h_dim, self.baseline_h_dim])) #GCNConv(self.baseline_h_dim, self.baseline_h_dim)
                self.GCNConv_out = GINConv(MLP([self.baseline_h_dim, self.baseline_h_dim, self.label_dim])) #GCNConv(self.baseline_h_dim, self.label_dim)

            def forward(self, feature, edge_index):
                x = self.activation(self.GCNConv_in(feature, edge_index))
                for i in range(self.n_layer - 2):
                    x = self.activation(self.GCNConv_hidden(x, edge_index))
                y_hat = nn.Softmax(dim=1)(self.GCNConv_out(x, edge_index))
                return y_hat

        baseline_model = BaselineGCN(self)
        return baseline_model

    # ------------------------------------------------------------------------------#

    def train(self, data):
        x = data.x
        edge_index = data.edge_index
        y = data.y
        unique_edge_index = data.unique_edge_index
        losses = []
        acc = []
        for i in range(self.iteration):

            # Generate  selection probability
            self.actor.eval()
            with torch.no_grad():
                z = self.actor.encode(x, edge_index)
                edge_selection_probability = self.actor.decode(z, unique_edge_index)

            unique_edge_selection = bernoulli_sampling_edges(edge_selection_probability)
            unique_edge_selection = torch.from_numpy(unique_edge_selection)

            edge_selection = unique_edge_selection.repeat(1, 2).view(-1)
            # Update weights (keras_ train_on_batch)
            # Critic loss
            self.critic.train()
            self.critic.optimizer.zero_grad()
            # Forward pass
            self.critic.output = self.critic(x, edge_index, edge_selection)
            self.critic.loss_value = F.cross_entropy(self.critic.output[self.train_mask], y[self.train_mask])

            # Backward pass
            self.critic.loss_value.backward()
            # Update weights
            self.critic.optimizer.step()
            # Megnezzuk a kimenetet (predict)
            self.critic.eval()
            with torch.no_grad():
                self.critic.output = self.critic(x, edge_index, edge_selection)

            # Baseline output
            if self.model_type == 'invase':
                self.baseline.train()
                # Baseline loss
                self.baseline.optimizer.zero_grad()
                # Forward pass
                self.baseline.output = self.baseline(x, edge_index)

                self.baseline.loss_value = F.cross_entropy(self.baseline.output[self.train_mask],
                                                           y[self.train_mask])
                # Backward pass
                self.baseline.loss_value.backward()
                # Update weights
                self.baseline.optimizer.step()
                self.baseline.eval()
                with torch.no_grad():
                    self.baseline.output = self.baseline(x, edge_index)

            # Train actor
            # Use multiple things as the y_true:
            # - selection, critic_out, baseline_out, and ground truth (y)
            if self.model_type == 'invase':
                y_final = torch.cat((x, self.critic.output.detach(),
                                     self.baseline.output.detach()), dim=1)  # axis=1
            else:  # invase_minus
                y_final = torch.cat((x, self.critic.output.detach()), dim=1)
                # Train the actor
            self.actor.train()
            self.actor.optimizer.zero_grad()
            # Forward pass

            z = self.actor.encode(x, edge_index)
            edge_selection_probability = self.actor.decode(z, unique_edge_index)

            self.actor.loss_value = self.actor.loss_invase(y_final, y, unique_edge_selection, edge_selection_probability)

            # Backward pass
            self.actor.loss_value.backward()
            # Update weights
            self.actor.optimizer.step()

            # Print the progress
            if self.model_type == 'invase':
                dialog = 'Iterations: ' + str(i) + \
                         ', critic loss: ' + str(self.critic.loss_value) + \
                         ', baseline loss: ' + str(self.baseline.loss_value) + \
                         ', actor loss: ' + str(self.actor.loss_value)
            else:  # self.model_type == 'invase_minus':
                dialog = 'Iterations: ' + str(i) + \
                         ', critic loss: ' + str(self.critic.loss_value) + \
                         ', actor loss: ' + str(self.actor.loss_value)
            if i % 250 == 0:
                # print('Iterations: ' + str(i) + "  Baseline and Critic test:")
                acc.append(self.test(data, self.train_mask, self.test_mask))
                print("Number of important edges: ", unique_edge_selection.sum())

                print(dialog)
                print(self.evaluate(data))
            losses.append([self.critic.loss_value.detach().numpy(), self.baseline.loss_value.detach().numpy(),
                           self.actor.loss_value.detach().numpy()])

        # Plot the losses
        plt.plot(losses)
        plt.legend(['critic loss', 'baseline loss', 'actor loss'])

        plt.savefig('Loss.png')


        plt.plot(acc)
        plt.legend(['baseline test acc', 'baseline train acc', 'critic test acc', 'critic train acc'])

        plt.savefig('Acc.png')

    def importance_score(self, data, important_edge_th):
        self.actor.eval()
        with torch.no_grad():
            z = self.actor.encode(data.x, data.edge_index)
            edge_selection_probability = self.actor.decode(z, data.unique_edge_index)

        #edge_selection = bernoulli_sampling_edges(edge_selection_probability)
        #edge_selection = torch.from_numpy(edge_selection)


        edge_selection = edge_selection_probability > 0.5 # threshold_value
        # Generate a node importance score from masks
        # A_edges = edge_index[:, A_mask].t().numpy()
        # A_nodes = A_edges[0] + A_edges[1]
        # A_nodes = np.unique(A_nodes)
        return edge_selection

    @torch.no_grad()
    def test(self, data, train_mask, test_mask):
        x = data.x
        edge_index = data.edge_index
        y = data.y
        # Test the baseline model
        self.baseline.eval()
        with torch.no_grad():
            pred = self.baseline(x, edge_index).argmax(dim=-1)

        train_correct = int((pred[train_mask] == y[train_mask]).sum())
        train_acc = train_correct / train_mask.size(0)

        test_correct = int((pred[test_mask] == y[test_mask]).sum())
        test_acc = test_correct / test_mask.size(0)

        # Test the critic model
        # Generate  selection probability

        self.actor.eval()
        with torch.no_grad():
            z = self.actor.encode(x, edge_index)
            edge_selection_probability = self.actor.decode(z, data.unique_edge_index)

        print("min", edge_selection_probability.max())
        print("max", edge_selection_probability.min())
        sorted_tensor = np.sort(edge_selection_probability)
        index_40th_percentile = int(0.6 * len(edge_selection_probability))
        threshold_value = sorted_tensor[index_40th_percentile]

       # edge_selection = edge_selection_probability > threshold_value
        edge_selection = edge_selection_probability > 0.5
        edge_selection = edge_selection.repeat(1, 2).view(-1)
        self.critic.eval()
        with torch.no_grad():
            pred = self.critic(x, edge_index, edge_selection).argmax(dim=-1)
            c_train_correct = int((pred[train_mask] == y[train_mask]).sum())
            c_train_acc = c_train_correct / train_mask.size(0)

            c_test_correct = int((pred[test_mask] == y[test_mask]).sum())
            c_test_acc = c_test_correct / test_mask.size(0)
        #  a = x_selection.size()
        # Print the progress
        # print("Selected nodes: ") + str(x_selection.sum()) + " out of "
        # print(x_selection.shape[1])
        #  print("Selected edges: ") + str((A_mask == True).sum()) + " out of " + str(A_mask.shape[1])
        dialog = 'BASELINE:  Test acc: ' + str(test_acc) + ' train acc: ' + str(
            train_acc) + '\n CRITIC:  Test acc: ' + str(c_test_acc) + ' train acc: ' + str(c_train_acc)

        print(dialog)
        return [test_acc, train_acc, c_test_acc, c_train_acc]

    def evaluate(self, data, important_node_th=0.5, important_edge_th=0.5):
        baseline_pred, y_hat, y_hat_inverse, edge_selection, edge_selection_probability = self.predict(data, important_edge_th)
        # Acc of the original labels
        correct_baseline = int((baseline_pred[self.test_mask] == data.y[self.test_mask]).sum())
        baseline_acc = correct_baseline / self.test_mask.size(0)
        # Acc of the graph only with the important edges
        correct_y_hat = int((y_hat[self.test_mask] == data.y[self.test_mask]).sum())
        acc_important_edges = correct_y_hat / self.test_mask.size(0)
        # Acc of the graph without the important edges
        correct_y_hat_inverse = int((y_hat_inverse[self.test_mask] == data.y[self.test_mask]).sum())
        acc_unimportant_edges = correct_y_hat_inverse / self.test_mask.size(0)
        Sparsity = 1 - (edge_selection.sum() / data.edge_index.size(1))
        print("Sparsity:", Sparsity, "from:", data.edge_index.size(1) / 2, " ", edge_selection.sum() / 2)
        print("ACC of the whole graph: ", baseline_acc, ",\n ACC with only the important edges", acc_important_edges,
              "\n ACC with only the unimportant edges", acc_unimportant_edges)

        # Fidelity score
        Fidelity_plus = baseline_acc - acc_unimportant_edges
        Fidelity_minus = baseline_acc - acc_important_edges
        print("Fidelity plus score: ", Fidelity_plus, "\nFidelity minus score: ", Fidelity_minus)

    def predict(self, data, important_node_th=0.5, important_edge_th=0.5):
        # Baseline_predictions
        baseline_pred = self.predict_baseline(data)

        torch.set_printoptions(threshold=5000)
        # Prediction for the whole graph
        A = data.edge_index
        x = data.x

        # Getting the importance masks
        self.actor.eval()
        with torch.no_grad():
            z = self.actor.encode(x, A)
            edge_selection_probability = self.actor.decode(z, data.unique_edge_index)

        # Sampling the features based on the selection_probability
        #edge_selection = bernoulli_sampling_edges(edge_selection_probability)
        #edge_selection = torch.from_numpy(edge_selection)

        # Select the edges that are above the threshold
        print('MAX: ', edge_selection_probability.max())
        print('MIN: ', edge_selection_probability.min())

        # Step 1: Sort the tensor in ascending order
        # sorted_tensor = np.sort(edge_selection_probability)

        # Step 2: Find the threshold value
        # Calculate the index for the 40th percentile
       # index_40th_percentile = int(0.6 * len(edge_selection_probability))

        # Find the threshold value
        #threshold_value = sorted_tensor[index_40th_percentile]


        edge_selection = edge_selection_probability > 0.5 #threshold_value
        edge_selection = edge_selection.repeat(1, 2).view(-1)


        #edge_selection = edge_selection_probability > 0.5

        # print(edge_selection[   0:10])
        # print(edge_selection[1980:1990])
        # randomly
        # edge_selection = torch.randint(low=0, high=2, size=edge_selection.shape)

        # Critic prediction - with Importance masks
        self.critic.eval()
        with torch.no_grad():
            y_hat = self.critic(x, A, edge_selection).argmax(dim=-1)

        # Critic prediction - with Unimportance masks
        edge_selection_inverse = (~(edge_selection.to(torch.bool)))
        edge_selection_inverse = edge_selection_inverse.to(torch.int64)
        self.critic.eval()
        with torch.no_grad():
            y_hat_inverse = self.critic(x, A, edge_selection_inverse).argmax(dim=-1)

        return baseline_pred, y_hat, y_hat_inverse, edge_selection, edge_selection_probability

    def predict_baseline(self, data):
        x = data.x
        edge_index = data.edge_index
        # Test the baseline model
        self.baseline.eval()
        with torch.no_grad():
            pred = self.baseline(x, edge_index).argmax(dim=-1)
        # correct = int((pred == data.y).sum())
        # acc = correct / data.num_nodes

        return pred

    def show_graphs(self, data, A_mask, X_mask, params, A_selection):
        if params['node_graph']:
            indexes = [i for i, value in enumerate(X_mask) if value]
            Node_graph = data.subgraph(torch.tensor([indexes])[0])
            Node_graph = to_networkx(Node_graph)
            visualize(data.y[torch.tensor([indexes])], Node_graph, "important_nodes")

        if params['edge_graph']:
            print(A_mask.sum())

            bool_mask = [bool(item) for item in A_mask]
            Edge_graph = nx.from_edgelist(data.edge_index[:, bool_mask].t().numpy())
            edge_labels = data.y[list(Edge_graph.nodes)]
            print(edge_labels)
            #print(A_selection)
            visualize(edge_labels, Edge_graph, "important_edges")

            # all_edges_list = data.edge_index.t().numpy()
            # all_labels_list = data.y.tolist()

            '''Nem fontos élek megjelenítése
             resultant_edges = [tuple(edge) for edge in all_edges if
                              tuple(edge) not in [tuple(masked_edge) for masked_edge in
                                                   data.edge_index[:, A_mask].t().numpy()]]
             G = nx.from_edgelist(resultant_edges)               
            #edge_labels = data.y[list(G.nodes)] 
            # # visualize(edge_labels, G,"nem fontos élek")
            '''

            # for i in range(len(all_labels_list) - 1):
            #    if i in test_mask_list:
            #        node_labels[i] = 1
            #    else:
            #        node_labels[i] = 0
            # Create a dictionary to map node indices to their labels.
        # node_label_dict = {node_idx: label for node_idx, label in enumerate(node_labels)}
        # Assign labels to nodes in the subgraph based on the dictionary.
        # for node in Edge_graph.nodes():
        #   Edge_graph.nodes[node]['label'] = node_label_dict[node]
        # Get the labels of the nodes in the subgraph.
        # a = nx.get_node_attributes(H, 'label')

        if params['metszet']:
            intersection_edges = list(set(Edge_graph.edges()) & set(Node_graph.edges()))
            intersection_graph = nx.Graph()
            intersection_graph.add_edges_from(intersection_edges)
            node_labels_list = [1 for node in intersection_graph.nodes]
            visualize(node_labels_list, intersection_graph, "intersection")

        if params['unio']:
            union_graph = nx.Graph()
            union_graph.add_edges_from(Edge_graph.edges())
            union_graph.add_edges_from(Node_graph.edges())
            node_labels_list = [1 for node in union_graph.nodes]
            visualize(node_labels_list, union_graph, "unio")

        if params['k_hop_subgraph']:
            test_mask_list = self.test_mask.tolist()
            new = []
            for i in test_mask_list:
                if data.y[i] != 0:
                    new.append(i)
            test_mask_list = new
            a, b = data.edge_index
            reverse_edges = torch.stack([b, a], dim=0)
            k_hops_edges_indexes = data.edge_index
            r = random.randint(0, len(test_mask_list))
            (k_hop_subset, k_hop_edge_index, k_hop_mapping, k_hop_edge_mask) = k_hop_subgraph(
                node_idx=test_mask_list[r], \
                num_hops=3, edge_index=k_hops_edges_indexes, \
                relabel_nodes=False, num_nodes=int(data.num_nodes), directed=False)
            #print(data.num_nodes)
            #print(k_hop_subset)
            #print(k_hop_edge_index)

            k_hop_graph = nx.from_edgelist(k_hop_edge_index.t().numpy())
            k_hop_graph_nodes = list(k_hop_graph.nodes)
            test_color = data.y.clone()
            test_color[test_mask_list[r]] = max(data.y) + 4
            # for i in test_mask_list:
            #     test_color[i] = max(data.y) + 4
            k_hop_graph_labels = test_color[k_hop_graph_nodes]

            #visualize(k_hop_graph_labels, k_hop_graph, "k_hops_from_testdata")

            all_edges_list = data.edge_index.t().numpy()
            undirected_edges = []
            colors = []
            # Get the A_mask and the edge colors for the undirected nx_graph
            for i in range(all_edges_list.shape[0]):
                if all_edges_list[i] in k_hop_graph.edges:
                    if {all_edges_list[i, 0], all_edges_list[i, 1]} not in undirected_edges:
                        undirected_edges.append({all_edges_list[i, 0], all_edges_list[i, 1]})
                        if A_mask[i]:
                            colors.append("blue")
                        else:
                            colors.append("red")
                    else:
                        if A_mask[i]:
                            colors[undirected_edges.index({all_edges_list[i, 0], all_edges_list[i, 1]})] = "blue"

            # Sort the edges and colors
            paired_list = list(zip(undirected_edges, colors))
            sorted_paired_list = sorted(paired_list, key=lambda pair: min(pair[0]))
            undirected_edges, colors = zip(*sorted_paired_list)

            visualize(k_hop_graph_labels, k_hop_graph, "k_hops_from_a_random_point", colors)

        if params['whole_graph']:
            all_edges_list = data.edge_index.t().numpy()
            undirected_edges = []
            colors = []

            A_selection = A_selection.tolist()
            c = [(w - min(A_selection)) / (max(A_selection) - min(A_selection)) for w in A_selection]

            # Get the A_mask and the edge colors for the undirected nx_graph
            for i in range(all_edges_list.shape[0]):
                if {all_edges_list[i, 0], all_edges_list[i, 1]} not in undirected_edges:
                    undirected_edges.append({all_edges_list[i, 0], all_edges_list[i, 1]})
                    c.append(A_selection[i])
                    if A_mask[i]:
                        colors.append("blue")
                    else:
                        colors.append("red")
                else:
                    if A_mask[i]:
                        colors[undirected_edges.index({all_edges_list[i, 0], all_edges_list[i, 1]})] = "blue"

            # Sort the edges and colors
            paired_list = list(zip(undirected_edges, colors, c))
            sorted_paired_list = sorted(paired_list, key=lambda pair: min(pair[0]))
            undirected_edges, colors, c = zip(*sorted_paired_list)

            Graph = to_networkx(data)
            # visualize(data.y, Graph, "Élek fontossága a teljes gráfban:", c)
            visualize(data.y, Graph, "whole_graf", colors)
