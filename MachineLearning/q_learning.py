from MachineLearning.neural_network_classes import NeuralNetwork, ConnectedLayer, GradientDescent, MomentumGradientDescent
from MachineLearning.activation_functions import relu, linear, tanh, sigmoid, softmax, leaky_relu

import numpy as np
import random
import datetime
import copy
import json

from matplotlib import pyplot

import global_settings as gs


class DeepQLearning:
    """
    Deep Q Learning Class
    Child classes implement the neural network
    Support for
        epsilon greedy
        target networks
        save/load
    """
    def __init__(self, state_n, actions_n, load=None):
        self.state_n = state_n
        self.actions_n = actions_n

        # number of frames since start
        self.frame_num = 0

        settings = gs.Q_LEARNING_SETTINGS

        # hyperparameters copied from global settings
        self.learning_rate = settings["LEARNING_RATE"]
        self.discount_rate = settings["DISCOUNT_RATE"]
        self.epsilon = settings["EPSILON_PROBABILITY"] if settings["TRAINING"] else 0
        self.epsilon_decay = settings["EPSILON_DECAY"]
        self.min_epsilon = settings["EPSILON_MIN"]
        self.network_copy_steps = settings["TARGET_NET_COPY_STEPS"]
        self.gd_momentum = settings["GD_MOMENTUM"]

        gd = GradientDescent if self.gd_momentum is None else MomentumGradientDescent

        # load model if needed, or create a new one
        if load is not None:
            self.network: NeuralNetwork = self.load_model(load)
            # update net with new LR
            self.network.learning_rate = self.learning_rate
            # update net with new gradient descent algorithm
            for l in self.network.layers:
                l.set_gradient_descent(gd, momentum=self.gd_momentum)

            print("LOADED: ", load)
        else:
            net_kwargs = {"gradient_descent": gd, "gd_momentum": self.gd_momentum}
            self.network = self.create_network(**net_kwargs)

        # target network is a copy of the normal network, with static weights + bias
        self.target_network = copy.deepcopy(self.network)

        # a list of experiences per episode
        self.experience_buffer = []
        self.train_amount = settings["TRAIN_AMOUNT"]  # fraction of experiences to train on
        self.max_buffer_length = settings["BUFFER_LENGTH"]

        self.reward_cache = []
        self.error_cache = []

    def create_network(self, **kwargs):
        raise NotImplementedError

    def decay_exploration_probability(self):
        # Decrease exploration exponentially
        # y = e^(-decay*x)
        # so
        # new = old * e^-decay
        self.epsilon = max(self.epsilon * np.exp(-self.epsilon_decay), self.min_epsilon)

    def get_action(self, state):
        # get action for state (largest q value)
        # if probability is correct, choose random action (epsilon greedy)
        if random.random() < self.epsilon and gs.TRAINING:
            action = random.randint(0, self.actions_n-1)
            return action, [int(a == action) for a in range(self.actions_n)]

        q_values = self.get_q_values(state)

        return q_values.index(max(q_values)), q_values

    def get_q_values(self, state, target=False):
        raise NotImplementedError

    def update_experience_buffer(self, state, action, reward):
        self.experience_buffer.append((tuple(state), action, reward))

        if len(self.experience_buffer) > self.max_buffer_length:
            self.experience_buffer.pop(0)

    def update_target_network(self):
        # deepcopy keeps no references etc. to old values
        self.target_network = copy.deepcopy(self.network)

    def fit(self, state, correct_q_values):
        raise NotImplementedError

    def train(self, verbose):
        # number of experiences to train on
        training_experiences_count = int(len(self.experience_buffer) * self.train_amount) - 1

        if training_experiences_count > 0:

            if verbose >= 2:
                print("============================================")

            # sample indicies (in experience buffer) of experiences to train on, randomly
            experiences_indices = random.sample(range(len(self.experience_buffer)), training_experiences_count)

            total_error = 0

            for experience_num in experiences_indices:
                # experience: (state, action, reward)
                experience = self.experience_buffer[experience_num]

                try:
                    next_experience = self.experience_buffer[experience_num + 1]
                except IndexError:
                    next_experience = None

                state = experience[0]
                action = experience[1]

                # reward gained from taking the action
                reward = experience[2]

                if next_experience is not None:
                    max_next_q_value = max(self.get_q_values(next_experience[0], target=True))
                else:
                    max_next_q_value = 0

                # bellman optimal equation
                q_target = reward + (self.discount_rate * max_next_q_value)

                # correct q values
                correct_q_values = self.get_q_values(state)

                if self.frame_num % 10 == 0 and verbose >= 3:
                    # dont print all Q values, only about 1/10 to reduce printed text
                    print("Q VALUES", correct_q_values)
                if experience_num == len(self.experience_buffer)-1 and verbose >= 3:
                    # also print final Q values as it can be useful
                    print("Q VALUES FINAL", correct_q_values)

                # if predicted Q values were (0.1, 0.2, 0.3)
                # and action [1] was taken
                # correct q values are (0.1, q_target, 0.3)

                correct_q_values[action] = q_target

                error = self.fit(state, correct_q_values)
                total_error += error

                # copy target network every network_copy_steps
                self.frame_num += 1
                if self.frame_num % self.network_copy_steps == 0:
                    self.update_target_network()

            if verbose >= 2:
                print("Exploration:", self.epsilon)
                print("Reward:", sum(map(lambda x: x[2], self.experience_buffer)))
                print("============================================\n")

            # sum of rewards in the episode
            self.reward_cache.append(sum(map(lambda x: x[2], self.experience_buffer)))
            self.error_cache.append(total_error/training_experiences_count)

            self.experience_buffer = []

            if verbose >= 3:
                # a method of troubleshooting by predicting a state to see how large changes are
                test_q_values = self.get_q_values(([30, 35, 43, 55, 300, 55, 43, 35, 30, 3]*100)[:self.state_n])
                print("TEST Q VALUES", test_q_values)

    def reward_graph(self, **kwargs):
        print("LEARNING RATE:", self.learning_rate)
        pyplot.plot(self.reward_cache, **kwargs)
        pyplot.show()

    @property
    def mean_rewards(self):
        return int(sum(self.reward_cache) / len(self.reward_cache))

    def error_graph(self, **kwargs):
        pyplot.plot(self.error_cache, **kwargs)
        pyplot.show()

    def save_model(self, type):
        raise NotImplementedError

    def _save_settings(self, path: str):
        # path is the file path and name of saved model
        path += "_settings.txt"
        with open(path, "w") as file:
            file.write(json.dumps(gs.Q_LEARNING_SETTINGS, separators=(", ", ":")).replace(" ", "\n"))

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError


"""
======= Individual Neural Network types =======
"""


class CustomModelQLearning(DeepQLearning):
    """
    Uses Custom model as Neural Network in Deep Q Learning
    """
    def create_network(self, **kwargs):
        return NeuralNetwork(
            [
                ConnectedLayer(relu, self.state_n, 12),
                ConnectedLayer(relu, 12, 18),
                ConnectedLayer(relu, 18, 36),
                ConnectedLayer(relu, 36, 50),
                ConnectedLayer(relu, 50, 50),
                ConnectedLayer(relu, 50, 36),
                ConnectedLayer(relu, 36, 18),
                ConnectedLayer(relu, 18, 9),
                ConnectedLayer(linear, 9, self.actions_n)
            ], learning_rate=self.learning_rate, **kwargs)

    def get_q_values(self, state, target=False):
        net = self.target_network if target else self.network
        return net.predict(state).tolist()

    def fit(self, state, correct_q_values):
        # return error
        return self.network.train([state], [correct_q_values], epochs=1, log=False)[0]

    def save_model(self, type):
        # time and average of rewards
        name = gs.SAVED_MODELS_ROOT + type + "_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(self.mean_rewards)
        self.network.save_to_file(name)
        self._save_settings(name)
        pyplot.plot(self.reward_cache)
        pyplot.savefig(name+".jpg")

    @classmethod
    def load_model(cls, name):
        return NeuralNetwork.load_from_file(gs.SAVED_MODELS_ROOT + name)
