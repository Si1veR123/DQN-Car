from MachineLearning.neural_network_classes import NeuralNetwork, ConnectedLayer
from MachineLearning.activation_functions import relu, linear, tanh, sigmoid, softmax, leaky_relu

import os
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

import global_settings

LOAD_MODEL = r"E:\EPQ\PythonAI\saved_models\custom_model_22.07;22.52_328"


class QLearning:
    def __init__(self, state_n, actions_n):
        self.state_n = state_n
        self.actions_n = actions_n

        self.frame_num = 0

        # hyperparameters copied from global settings
        self.learning_rate = global_settings.LEARNING_RATE
        self.discount_rate = global_settings.DISCOUNT_RATE
        self.exploration_probability = global_settings.EXPLORATION_PROBABILITY
        self.exploration_decay = global_settings.EXPLORATION_DECAY
        self.network_copy_steps = global_settings.TARGET_NET_COPY_STEPS

        if LOAD_MODEL:
            self.network = self.load_model(LOAD_MODEL)
            print("LOADED: ", LOAD_MODEL.split("\\")[-1])
        else:
            self.network = self.create_network()

        self.target_network = copy.deepcopy(self.network)

        self.experience_buffer = []
        self.train_amount = 0.7  # fraction of experiences to train on

        self.reward_cache = []

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

    def decay_exploration_probability(self):
        self.exploration_probability = self.exploration_probability * np.exp(-self.exploration_decay)

    def get_action(self, state):
        if random.random() < self.exploration_probability:
            action = random.randint(0, self.actions_n-1)
            return action, [int(a == action) for a in range(self.actions_n)]

        q_values = self.get_q_values(state)

        return q_values.index(max(q_values)), q_values

    def get_q_values(self, state, target=False):
        net = self.target_network if target else self.network
        return net(np.array([state]))[0].tolist()

    def update_experience_buffer(self, state, action, reward):
        self.experience_buffer.append((tuple(state), action, reward))

        if len(self.experience_buffer) > 5000:
            self.experience_buffer.pop(0)

    def update_target_network(self):
        print("=========UPDATING TARGET NETWORK=========")
        self.target_network = copy.deepcopy(self.network)

    def fit(self, state, correct_q_values):
        self.network.fit(np.array([state]), np.array([correct_q_values]))

    def train(self):
        if len(self.experience_buffer):
            print("============================================")
            training_experiences_count = int(len(self.experience_buffer) * self.train_amount) - 1
            experiences_indicies = random.sample(range(len(self.experience_buffer) - 1), training_experiences_count)

            for experience_num in experiences_indicies:
                # experience: (state, action, reward)
                experience = self.experience_buffer[experience_num]
                next_experience = self.experience_buffer[experience_num + 1]

                state = experience[0]
                action = experience[1]
                reward = next_experience[2]

                max_next_q_value = max(self.get_q_values(next_experience[0], target=True))

                # bellman optimal equation
                q_target = reward + (self.discount_rate * max_next_q_value)

                # correct q values
                q_values = self.get_q_values(state)
                print("Q VALUES", q_values)

                correct_q_values = q_values
                correct_q_values[action] = q_target

                self.fit(state, correct_q_values)

                self.frame_num += 1
                if self.frame_num % self.network_copy_steps == 0:
                    self.update_target_network()

            print("Exploration:", self.exploration_probability)
            print("Reward:", sum(map(lambda x: x[2], self.experience_buffer)))
            print("============================================\n")

            self.reward_cache.append(sum(map(lambda x: x[2], self.experience_buffer)))

            self.experience_buffer = []

    def reward_graph(self):
        pyplot.plot(self.reward_cache)
        pyplot.show()

    def save_model(self, path):
        # time and average of rewards
        name = path + "keras_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(int(sum(self.reward_cache)/len(self.reward_cache)))

        self.network.save(name, include_optimizer=True)

    @classmethod
    def load_model(cls, path):
        return keras.models.load_model(path)


class CustomModelQLearning(QLearning):
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

    def save_model(self, path):
        # time and average of rewards
        name = path + "custom_model_" + datetime.datetime.now().strftime("%d.%m;%H.%M") + "_" + str(int(sum(self.reward_cache) / len(self.reward_cache)))
        self.network.save_to_file(name)

    @classmethod
    def load_model(cls, path):
        return NeuralNetwork.load_from_file(path)
