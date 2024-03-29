# -*- coding: utf-8 -*-

# Necessary packages

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from INVASE_MLP.Utilites_for_MLP import *

import torch
from torch import nn

"""INVASE_PyTorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x669fb3yPt7XAgaRaKIY7Y4ToB-a86SG

# **INVASE algorithm implementation in Pytorch**

Based on

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "IINVASE: Instance-wise Variable Selection using Neural Networks," 

           International Conference on Learning Representations (ICLR), 2019.

Paper Link: https://openreview.net/forum?id=BJg_roAcK7


---
"""


# **INVASE Model**
class CrossEntropy():
    def __init__(self, reduction=None):
        self.reduction = reduction

    def __call__(self, y_ground_truth, out):
        result = -torch.sum(y_ground_truth * torch.log(out + 1e-8), dim=1)
        if self.reduction == 'mean':
            result = torch.mean(result)
        return result


class Invase():

    def __init__(self, x_train, y_train, model_type, model_parameters):

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
        self.label_dim = y_train.shape[1]
        self.model_type = model_type

        # Build and compile critic
        self.critic = self.build_critic()
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        self.critic.loss = CrossEntropy(reduction='mean')

        # Build and compile the actor
        self.actor = self.build_actor()
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.actor.loss = self.actor_loss

        if self.model_type == 'invase':
            # Build and compile the baseline
            self.baseline = self.build_baseline()
            self.baseline.optimizer = torch.optim.Adam(self.baseline.parameters(), self.learning_rate)
            self.baseline.loss = CrossEntropy(reduction='mean')

    def actor_loss(self, y_true, y_pred):
        '''
        y_true contains all outputs (actor, critic, basline) + ground truth
        '''
        # Actor output
        actor_out = y_true[:, :self.dim]
        # Critic output
        critic_out = y_true[:, self.dim:(self.dim + self.label_dim)]

        if self.model_type == 'invase':
            # Baseline output
            baseline_out = y_true[:, (self.dim + self.label_dim):(self.dim + 2 * self.label_dim)]
            # Ground truth label
            y_ground_truth = y_true[:, (self.dim + 2 * self.label_dim):]
        elif self.model_type == 'invase_minus':
            # Ground truth label
            y_ground_truth = y_true[:, (self.dim + self.label_dim):]

            # Critic loss
        critic_loss = CrossEntropy()(y_ground_truth,
                                     critic_out)  # -torch.sum(y_ground_truth * torch.log(critic_out + 1e-8), dim=1)

        if self.model_type == 'invase':
            # Baseline loss
            baseline_loss = CrossEntropy()(y_ground_truth,
                                           baseline_out)  # -torch.sum(y_ground_truth * torch.log(baseline_out + 1e-8), dim=1)
            # Reward
            Reward = -(critic_loss - baseline_loss)

        elif self.model_type == 'invase_minus':
            Reward = -critic_loss

        # Policy gradient loss computation. 

        custom_actor_loss = Reward * torch.sum(actor_out * torch.log(y_pred + 1e-8) + \
                                               (1 - actor_out) * torch.log(1 - y_pred + 1e-8), dim=1) - \
                            self.lamda * torch.mean(y_pred, dim=1)

        # Custom actor loss
        custom_actor_loss = torch.mean(-custom_actor_loss)

        return custom_actor_loss

    def build_actor(self):

        class Actor(nn.Module):
            def __init__(self, params):
                super(Actor, self).__init__()
                # Params
                self.activation = params.activation
                self.n_layer = params.n_layer
                self.dim = params.dim
                self.actor_h_dim = params.actor_h_dim

                # Layers
                self.linear_in = nn.Linear(self.dim, self.actor_h_dim)
                self.linear_hidden = nn.Linear(self.actor_h_dim, self.actor_h_dim)
                self.linear_out = nn.Linear(self.actor_h_dim, self.dim)

            def forward(self, feature):
                x = self.activation(self.linear_in(feature))
                for i in range(self.n_layer - 2):
                    x = self.activation(self.linear_hidden(x))
                selection_probability = torch.sigmoid(self.linear_out(x))
                return selection_probability

        actor_model = Actor(self)
        return actor_model

    def build_critic(self):

        class Critic(nn.Module):
            def __init__(self, params):
                super(Critic, self).__init__()
                # Params
                self.activation = params.activation
                self.n_layer = params.n_layer
                self.label_dim = params.label_dim
                self.dim1 = params.dim
                self.critic_h_dim = params.critic_h_dim

                # Layers
                self.linear_in = nn.Linear(self.dim1, self.critic_h_dim)
                self.batch_normalization1 = nn.BatchNorm1d(self.critic_h_dim, momentum=0.01)
                self.linear_hidden = nn.Linear(self.critic_h_dim, self.critic_h_dim)
                self.batch_normalization2 = nn.BatchNorm1d(self.critic_h_dim, momentum=0.01)
                self.linear_out = nn.Linear(self.critic_h_dim, self.label_dim)

            def forward(self, feature, selection):
                # Element wise multiplication
                critic_model_input = (feature * selection).float()
                x = self.activation(self.linear_in(critic_model_input))
                x = self.batch_normalization1(x)
                for i in range(self.n_layer - 2):
                    x = self.activation(self.linear_hidden(x))
                    x = self.batch_normalization2(x)
                y_hat = nn.Softmax(dim=1)(self.linear_out(x))
                return y_hat

        critic_model = Critic(self)
        return critic_model

    def build_baseline(self):

        class Baseline(nn.Module):
            def __init__(self, params):
                super(Baseline, self).__init__()
                # Params
                self.activation = params.activation
                self.n_layer = params.n_layer
                self.label_dim = params.label_dim
                self.dim = params.dim
                self.baseline_h_dim = params.critic_h_dim  # same as the critic

                # Layers
                self.linear_in = nn.Linear(self.dim, self.baseline_h_dim)
                self.batch_normalization1 = nn.BatchNorm1d(self.baseline_h_dim, momentum=0.01)
                self.linear_hidden = nn.Linear(self.baseline_h_dim, self.baseline_h_dim)
                self.linear_out = nn.Linear(self.baseline_h_dim, self.label_dim)
                self.batch_normalization2 = nn.BatchNorm1d(self.baseline_h_dim, momentum=0.01)

            def forward(self, feature):
                x = self.activation(self.linear_in(feature))
                x = self.batch_normalization1(x)
                for i in range(self.n_layer - 2):
                    x = self.activation(self.linear_hidden(x))
                    x = self.batch_normalization2(x)
                y_hat = nn.Softmax(dim=1)(self.linear_out(x))
                return y_hat

        baseline_model = Baseline(self)
        return baseline_model

    # ------------------------------------------------------------------------------#

    def train(self, x_train, y_train):

        for i in range(self.iteration):

            ## Train critic
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx, :]
            y_batch = y_train[idx, :]

            # Generate a batch of selection probability
            self.actor.eval()
            with torch.no_grad():
                selection_probability = self.actor(x_batch)

                # Sampling the features based on the selection_probability
            selection = bernoulli_sampling(selection_probability)

            # Update weights (keras_ train_on_batch)
            # Critic loss
            self.critic.train()
            self.critic.optimizer.zero_grad()
            # Forward pass
            self.critic.output = self.critic(x_batch, selection)
            self.critic.loss_value = self.critic.loss(y_batch, self.critic.output)
            # Backward pass
            self.critic.loss_value.backward()
            # Update weights
            self.critic.optimizer.step()
            ##Megnezzuk a kimenetetn(predict)
            self.critic.eval()
            with torch.no_grad():
                self.critic.output = self.critic(x_batch, selection)

            # Baseline output
            if self.model_type == 'invase':
                self.baseline.train()
                # Baseline loss
                self.baseline.optimizer.zero_grad()
                # Forward pass
                self.baseline.output = self.baseline(x_batch)
                self.baseline.loss_value = self.baseline.loss(y_batch, self.baseline.output)
                # Backward pass
                self.baseline.loss_value.backward()
                # Update weights
                self.baseline.optimizer.step()
                self.baseline.eval()
                with torch.no_grad():
                    self.baseline.output = self.baseline(x_batch)

            ## Train actor
            # Use multiple things as the y_true:
            # - selection, critic_out, baseline_out, and ground truth (y_batch)
            if self.model_type == 'invase':

                y_batch_final = torch.cat((torch.from_numpy(selection), self.critic.output,
                                           self.baseline.output, y_batch), axis=1)
            elif self.model_type == 'invase_minus':
                y_batch_final = torch.cat((torch.from_numpy(selection), self.critic.output,
                                           y_batch), axis=1)
            # Train the actor
            self.actor.train()
            self.actor.optimizer.zero_grad()
            # Forward pass
            self.actor.output = self.actor(x_batch)
            self.actor.loss_value = self.actor.loss(y_batch_final.detach(), self.actor.output)
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
            elif self.model_type == 'invase_minus':
                dialog = 'Iterations: ' + str(i) + \
                         ', critic loss: ' + str(self.critic.loss_value) + \
                         ', actor loss: ' + str(self.actor.loss_value)
            if i % 1000 == 0:
                print(dialog)

    def importance_score(self, x):
        self.actor.eval()
        with torch.no_grad():
            feature_importance = self.actor(x)
        return np.asarray(feature_importance)

    def predict(self, x_test):

        # Generate a batch of selection probability
        self.actor.eval()
        with torch.no_grad():
            selection_probability = self.actor(x_test)
            # Sampling the features based on the selection_probability
        selection = bernoulli_sampling(selection_probability)
        # Prediction 
        self.critic.eval()
        with torch.no_grad():
            y_hat = self.critic(x_test, selection)
        return np.asarray(y_hat)
