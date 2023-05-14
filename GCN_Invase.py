# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Utilities import *
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from gala import GAE

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
    def __init__(self, x_train, A_train, y_train, model_type, model_parameters):
        self.lamda = model_parameters['lamda']
        self.actor_h_dim = model_parameters['actor_h_dim']
        self.critic_h_dim = model_parameters['critic_h_dim']
        self.n_layer = model_parameters['n_layer']
        self.batch_size = model_parameters['batch_size']
        self.iteration = model_parameters['iteration']
        if model_parameters['activation'] == "relu":
            self.activation = nn.ReLU()
        self.learning_rate = model_parameters['learning_rate']
        self.dim = x_train.shape[1]
        self.label_dim = y_train.shape[0]
        self.model_type = model_type

        # Build and compile critic
        self.critic = self.build_critic()
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        self.critic.loss = CrossEntropy(reduction='mean')

        # Build and compile the actor
        self.actor = self.build_actor(self)
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.actor.loss_invase = self.actor_loss

        if self.model_type == 'invase':
            # Build and compile the baseline
            self.baseline = self.build_baseline()
            self.baseline.optimizer = torch.optim.Adam(self.baseline.parameters(), self.learning_rate)
            self.baseline.loss = CrossEntropy(reduction='mean')

    def actor_loss(self, y_true, y,  y_pred):
        # y_true contains all outputs (actor, critic, baseline) + ground truth
        # Actor output
        actor_out = y_true[:, :self.dim]
        # Critic output
        critic_out = y_true[:, self.dim:(self.dim + self.label_dim)]

        if self.model_type == 'invase':
            # Baseline output
            baseline_out = y_true[:, (self.dim + self.label_dim):(self.dim + 2 * self.label_dim)]
            # Ground truth label
            y_ground_truth = y_true[:, (self.dim + 2 * self.label_dim):]
        else:  # self.model_type == 'invase_minus':
            # Ground truth label
            y_ground_truth = y_true[:, (self.dim + self.label_dim):]
        y_ground_truth = y
            # Critic loss
        critic_loss = CrossEntropy()(y_ground_truth, critic_out)

        if self.model_type == 'invase':
            # Baseline loss
            baseline_loss = CrossEntropy()(y_ground_truth, baseline_out)
            # Reward
            Reward = -(critic_loss - baseline_loss)

        else:  # self.model_type == 'invase_minus':
            Reward = -critic_loss

        # Policy gradient loss computation.

        custom_actor_loss = Reward * torch.sum(actor_out * torch.log(y_pred + 1e-8) +
                                               (1 - actor_out) * torch.log(1 - y_pred + 1e-8), dim=1) - \
                            self.lamda * torch.mean(y_pred, dim=1)

        # Custom actor loss
        custom_actor_loss = torch.mean(-custom_actor_loss)
        return custom_actor_loss

    def build_actor(self, params):
        # Params
        self.activation = params.activation
        self.n_layer = params.n_layer
        self.dim = params.dim
        # self.actor_h_dim = params.actor_h_dim

        # Autoencoder model
        actor_model = GAE(self.dim, round(self.dim/2), round(self.dim/4))
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
                self.GCNConv_in = GCNConv(self.dim1, self.critic_h_dim)
                self.batch_normalization1 = nn.BatchNorm1d(self.critic_h_dim, momentum=0.01)
                self.GCNConv_hidden = GCNConv(self.critic_h_dim, self.critic_h_dim)
                self.batch_normalization2 = nn.BatchNorm1d(self.critic_h_dim, momentum=0.01)
                self.GCN_Conv_out = GCNConv(self.critic_h_dim, self.label_dim)

            def forward(self, feature, edge_index, x_selection, A_selection):

                # Element wise multiplication
                # Use only the selected features and edges
                critic_model_input = (feature * x_selection).float()
                # TODO: ezt használni, ha A_selection már helyes
                # critic_model_A_input = (edge_index * A_selection).float()
                critic_model_A_input = edge_index
                x = self.activation(self.GCNConv_in(critic_model_input, critic_model_A_input))
                x = self.batch_normalization1(x)
                for i in range(self.n_layer - 2):
                    x = self.activation(self.GCNConv_hidden(x, critic_model_A_input))
                    x = self.batch_normalization2(x)
                y_hat = nn.Softmax(dim=1)(self.GCN_Conv_out(x, critic_model_A_input))
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
                self.GCNConv_in = GCNConv(self.dim, self.baseline_h_dim)
                self.batch_normalization1 = nn.BatchNorm1d(self.baseline_h_dim, momentum=0.01)
                self.GCNConv_hidden = GCNConv(self.baseline_h_dim, self.baseline_h_dim)
                self.GCNConv_out = GCNConv(self.baseline_h_dim, self.label_dim)
                self.batch_normalization2 = nn.BatchNorm1d(self.baseline_h_dim, momentum=0.01)

            def forward(self, feature, edge_index):
                x = self.activation(self.GCNConv_in(feature, edge_index))
                x = self.batch_normalization1(x)
                for i in range(self.n_layer - 2):
                    x = self.activation(self.GCNConv_hidden(x, edge_index))
                    x = self.batch_normalization2(x)
                y_hat = nn.Softmax(dim=1)(self.GCNConv_out(x, edge_index))
                return y_hat

        baseline_model = BaselineGCN(self)
        return baseline_model

    # ------------------------------------------------------------------------------#

    def train(self, x_train, A_train, y_train):

        for i in range(self.iteration):

            # Select a random batch of samples
            # TODO: Kellenek-e batchek? Jeleneg nem használom őket, de lehet, hogy kellene
            # ignore batches
            #  idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train # [idx, :]
            y_batch = y_train # [idx, :]

            # Generate  selection probability
            self.actor.eval()
            with torch.no_grad():
                # TODO: GALA encoder használatának javíítása
                # Itt az encodert és a dekódert is használom, hogy később a méretek stimmeljenek,
                # de úgy gondolom, hogy csak az enkóder kéne és abból valahogy vissza kéne állítani a többit
                z = self.actor.encode(x_batch, A_train)
                selection_probability = self.actor.decode(z, A_train)
                # Sampling the features based on the selection_probability
                selection_probability = torch.sigmoid(selection_probability)
            x_selection = bernoulli_sampling(selection_probability)
            # TODO: A_selection
            #A_selection = bernoulli_sampling(A_train)
            A_selection = A_train
            # Update weights (keras_ train_on_batch)
            # Critic loss
            self.critic.train()
            self.critic.optimizer.zero_grad()
            # Forward pass
            self.critic.output = self.critic(x_batch, A_train, x_selection, A_selection)
            self.critic.loss_value = self.critic.loss(y_batch, self.critic.output)
            # Backward pass
            self.critic.loss_value.backward()
            # Update weights
            self.critic.optimizer.step()
            # Megnezzuk a kimenetet(predict)
            self.critic.eval()
            with torch.no_grad():
                self.critic.output = self.critic(x_batch, A_train, x_selection, A_selection)

            # Baseline output
            if self.model_type == 'invase':
                self.baseline.train()
                # Baseline loss
                self.baseline.optimizer.zero_grad()
                # Forward pass
                self.baseline.output = self.baseline(x_batch, A_train)
                self.baseline.loss_value = self.baseline.loss(y_batch, self.baseline.output)
                # Backward pass
                self.baseline.loss_value.backward()
                # Update weights
                self.baseline.optimizer.step()
                self.baseline.eval()
                with torch.no_grad():
                    self.baseline.output = self.baseline(x_batch, A_train)

            # Train actor
            # Use multiple things as the y_true:
            # - selection, critic_out, baseline_out, and ground truth (y_batch)
            if self.model_type == 'invase':
                y_batch_final = torch.cat((torch.from_numpy(x_selection), self.critic.output,
                                           self.baseline.output), dim=1)  # axis=1
            else:  # invase_minus
                y_batch_final = torch.cat((torch.from_numpy(x_selection), self.critic.output), dim=1)
                # Train the actor
            self.actor.train()
            self.actor.optimizer.zero_grad()
            # Forward pass
            z = self.actor.encode(x_batch, A_train)
            self.actor.output = self.actor.decode(z, A_train)
            # TODO: Loss számításba belevenni a GALA loss-ját is, ne csak az INVASE actor loss legyen
            self.actor.loss_value = self.actor.loss_invase(y_batch_final.detach(), y_batch, self.actor.output)
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
            if i % 100 == 0:
                print(dialog)

    def importance_score(self, x):
        self.actor.eval()
        with torch.no_grad():
            feature_importance = self.actor(x)
        return np.asarray(feature_importance)

    def predict(self, x_test, A):

        # Generate a batch of selection probability
        self.actor.eval()
        with torch.no_grad():
            z = self.actor.encode(x_test, A)
            selection_probability = self.actor.decode(z, A)
            # Sampling the features based on the selection_probability
        x_selection = bernoulli_sampling(selection_probability)
        # TODO: A_selection kiszámítása z-ből
        A_selection = A
        # Prediction
        self.critic.eval()
        with torch.no_grad():
            y_hat = self.critic(x_test, x_selection, A_selection)
        return np.asarray(y_hat)
