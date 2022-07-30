from MachineLearning.neural_network_classes import NeuralNetwork, ConnectedLayer
from MachineLearning.activation_functions import relu, linear, tanh, sigmoid, softmax, leaky_relu

import os
# no gpu as it made it slower
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from keras.optimizers import Adam
import keras

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import numpy as np
import random
import datetime
import copy

from matplotlib import pyplot

import global_settings as gs


class QLearning:
    """
    Deep Q Learning Class
    Child classes implement the neural network
    Support for
        epsilon greedy
        target networks
        save/load
    """
    def __init__(self, state_n, actions_n):
        self.state_n = state_n
        self.actions_n = actions_n

        # number of frames since start
        self.frame_num = 0

        # hyperparameters copied from global settings
        self.learning_rate = gs.LEARNING_RATE
        self.discount_rate = gs.DISCOUNT_RATE
        self.exploration_probability = gs.EXPLORATION_PROBABILITY
        self.exploration_decay = gs.EXPLORATION_DECAY
        self.network_copy_steps = gs.TARGET_NET_COPY_STEPS

        # load model if needed, or create a new one
        if gs.LOAD_MODEL:
            self.network = self.load_model(gs.LOAD_MODEL)
            print("LOADED: ", gs.LOAD_MODEL)
        else:
            self.network = self.create_network()

        # target network is a copy of the normal network, with static weights + bias
        self.target_network = copy.deepcopy(self.network)

        # a list of experiences per episode
        self.experience_buffer = []
        self.train_amount = 0.7  # fraction of experiences to train on

        self.reward_cache = []

    def create_network(self):
        raise NotImplementedError

    def decay_exploration_probability(self):
        # Decrease exploration exponentially
        # y = e^(-decay*x)
        # so
        # new = old * e^-decay
        self.exploration_probability = self.exploration_probability * np.exp(-self.exploration_decay)

    def get_action(self, state):
        # get action for state (largest q value)
        # if probability is correct, choose random action (epsilon greedy)
        if random.random() < self.exploration_probability and gs.TRAINING:
            action = random.randint(0, self.actions_n-1)
            return action, [int(a == action) for a in range(self.actions_n)]

        q_values = self.get_q_values(state)

        return q_values.index(max(q_values)), q_values

    def get_q_values(self, state, target=False):
        raise NotImplementedError

    def update_experience_buffer(self, state, action, reward):
        self.experience_buffer.append((tuple(state), action, reward))

        if len(self.experience_buffer) > 5000:
            self.experience_buffer.pop(0)

    def update_target_network(self):
        print("=========UPDATING TARGET NETWORK=========")
        # deepcopy keeps no references etc. to old values
        self.target_network = copy.deepcopy(self.network)

    def fit(self, state, correct_q_values):
        raise NotImplementedError

    def train(self):
        if len(self.experience_buffer):
            print("============================================")
            # number of experiences to train on
            training_experiences_count = int(len(self.experience_buffer) * self.train_amount) - 1
            # seample indicies (in experience buffer) of experiences to train on, randomly
            experiences_indicies = random.sample(range(len(self.experience_buffer) - 1), training_experiences_count)

            for experience_num in experiences_indicies:
                # experience: (state, action, reward)
                experience = self.experience_buffer[experience_num]
                next_experience = self.experience_buffer[experience_num + 1]

                state = experience[0]
                action = experience[1]
                # get reward in next experience, as this is the
                # reward gained from taking the action in current experience
                reward = next_experience[2]

                max_next_q_value = max(self.get_q_values(next_experience[0], target=True))

                # bellman optimal equation
                q_target = reward + (self.discount_rate * max_next_q_value)

                # correct q values
                correct_q_values = self.get_q_values(state)
                print("Q VALUES", correct_q_values)

                # if predicted Q values were (0.1, 0.2, 0.3)
                # and action [1] was taken
                # correct q values are (0.1, q_target, 0.3)

                correct_q_values[action] = q_target

                self.fit(state, correct_q_values)

                # copy target network every network_copy_steps
                self.frame_num += 1
                if self.frame_num % self.network_copy_steps == 0:
                    self.update_target_network()

            print("Exploration:", self.exploration_probability)
            print("Reward:", sum(map(lambda x: x[2], self.experience_buffer)))
            print("============================================\n")

            # sum of rewards in the episode
            self.reward_cache.append(sum(map(lambda x: x[2], self.experience_buffer)))

            self.experience_buffer = []

    def reward_graph(self):
        pyplot.plot(self.reward_cache)
        pyplot.show()

    def save_model(self):
        raise NotImplementedError

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError


"""
======= Individual Neural Network types =======
"""

class CustomModelQLearning(QLearning):
    """
    Uses Custom model as Neural Network in Deep Q Learning
    """
    def create_network(self):
        return NeuralNetwork(
            [
                ConnectedLayer(relu, self.state_n, 12),
                ConnectedLayer(relu, 12, 18),
                ConnectedLayer(relu, 18, 9),
                ConnectedLayer(linear, 9, self.actions_n)
            ], learning_rate=self.learning_rate)

    def get_q_values(self, state, target=False):
        net = self.target_network if target else self.network
        return net.predict(state).tolist()

    def fit(self, state, correct_q_values):
        self.network.train([state], [correct_q_values], epochs=1, log=False)

    def save_model(self):
        # time and average of rewards
        name = gs.SAVED_MODELS_ROOT + "custom_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(int(sum(self.reward_cache) / len(self.reward_cache)))
        self.network.save_to_file(name)

    @classmethod
    def load_model(cls, name):
        return NeuralNetwork.load_from_file(gs.SAVED_MODELS_ROOT + name)


class KerasModelQLearning(QLearning):
    """
    Uses Keras Sequential model as Neural Network in Deep Q Learning
    """
    def create_network(self):
        network = Sequential([
            Input(shape=(self.state_n,)),
            Dense(12, activation="relu"),
            Dense(18, activation="relu"),
            Dense(9, activation="relu"),
            Dense(self.actions_n, activation="relu")
        ])

        network.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        return network

    def get_q_values(self, state, target=False):
        net = self.target_network if target else self.network
        return net(np.array([state]))[0].tolist()

    def fit(self, state, correct_q_values):
        self.network.fit(np.array([state]), np.array([correct_q_values]))

    def save_model(self):
        # time and average of rewards
        name = gs.SAVED_MODELS_ROOT + "keras_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(int(sum(self.reward_cache)/len(self.reward_cache)))

        self.network.save(name, include_optimizer=True)

    @classmethod
    def load_model(cls, path):
        return keras.models.load_model(gs.SAVED_MODELS_ROOT + path)
