from MachineLearning.activation_functions import relu

import numpy as np
import time
import pickle

"""
Classes to make a neural network using numpy arrays as matrices for weights and bias
"""


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate

    def reset(self):
        for l in self.layers:
            l.reset()

    def mse_error(self, predicted, expected, derivative=False):
        """
        Mean Squared Error function
        """
        if derivative:
            return 2 * (predicted - expected) / np.size(predicted)
        else:
            return np.mean(np.power((predicted - expected), 2))

    def predict(self, x) -> np.ndarray:
        values = np.array([x])

        if len(values.shape) >= 4:
            # batch
            predictions = []
            for value in values:
                running_value = value
                for layer in self.layers:
                    running_value = layer.forward(running_value)
                predictions.append(running_value)

        else:
            for layer in self.layers:
                values = layer.forward(values)

            return values[0]

    def train(self, x, y, epochs=10, log=True):
        """
        :param epochs: number of epochs to train for
        :param x:[
                    [inputs1],
                    [inputs2],
                    ...
                                ]
        :param y: [
                    [expected1],
                    [expected2],
                    ...
                                ]
        """

        assert len(x) == len(y)

        total_errors = []

        time_start = time.time()

        for epoch in range(epochs):

            epoch_error = 0
            for inputs, outputs in zip(x, y):
                # forward prop
                values = np.array([inputs])
                for layer in self.layers:
                    values = layer.forward(values)

                # backward prop
                errors = self.mse_error(values, outputs, derivative=True)

                for backwards_layer in self.layers[::-1]:
                    errors = backwards_layer.backward(errors, self.learning_rate)

                epoch_error += self.mse_error(values, outputs)

            total_errors.append(epoch_error/len(x))

            if log:
                print(f"Epoch: {epoch}\tError: {epoch_error/len(x)}\tEstimated time left: {((time.time() - time_start) / (epoch + 1)) * (epochs-(epoch+1))}")

        return total_errors

    def optimal_learning_rate(self, start, step, iterations, inputs, outputs, epochs=10):
        """
        Find optimal learning rate by training with different learning rates
        :param start: first learning rate
        :param step: increase in learning rate per iteration
        :param iterations: number of iterations
        :param inputs: training features
        :param outputs: training labels
        :param epochs: epochs per train
        :return: sorted dictionary by lr of {learning rate: error}
        """
        learning_rate = start
        learning_rate_error = {}

        for i in range(iterations):
            print(f"\n=======STARTING LEARNING RATE {learning_rate}=======\n")
            error = self.train(inputs, outputs, epochs=epochs)[-1]
            learning_rate_error[learning_rate] = error

            self.reset()
            learning_rate += step

        return dict(sorted(learning_rate_error.items(), key=lambda x:x[1]))

    def save_to_file(self, filename="saved_network.pickle"):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, filename="saved_network.pickle"):
        with open(filename, "rb") as file:
            obj = pickle.load(file)
        return obj


class Layer:
    def __init__(self, activation, input_size, output_size):
        self.input = np.empty(0)
        self.output = np.empty(0)

        if activation == relu:
            # He weight initialisation, used for relu
            self.weights = np.random.normal(0, np.sqrt(2/input_size), size=(input_size, output_size))
        else:
            # xavier weight initialisation

            lower, upper = -(1.0 / np.sqrt(input_size)), (1.0 / np.sqrt(input_size))
            self.weights = lower + np.random.rand(input_size, output_size) * (upper-lower)

        self.biases = np.full(shape=output_size, fill_value=0.1, dtype=np.float32)

        self.activation = activation

    def reset(self):
        a = self.activation
        input_size, output_size = self.weights.shape

        self.__init__(a, input_size, output_size)

    def forward(self, inputs):
        pass

    def backward(self, output_derivatives, learning_rate):
        pass


class ConnectedLayer(Layer):
    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.activation(self.output)

    def backward(self, output_derivatives, learning_rate):
        """
                                                                ↓ = output_derivatives
                ∂C/∂W (derivative of cost wrt weight) = Xᵀ * ∂C/∂Y
                ∂C/∂B (derivative of cost wrt bias) = ∂C/∂Y
                ∂C/∂X (input derivative) =  ∂C/∂Y * Wᵀ
                """

        undo_activation = np.multiply(output_derivatives, self.activation(self.output, derivative=True))

        input_d = np.dot(undo_activation, self.weights.T)
        d_weights = np.dot(self.input.T, undo_activation)

        self.weights = self.weights - d_weights * learning_rate
        self.biases = self.biases - undo_activation * learning_rate

        return input_d


class FlattenLayer(ConnectedLayer):
    def __init__(self, activation, output_size):
        super().__init__(activation=activation, input_size=1, output_size=output_size)

    def forward(self, inputs):
        return ConnectedLayer.forward(self, inputs=inputs)


class CustomConnected(ConnectedLayer):
    def __init__(self, activation, input_size, output_size, connections):
        super().__init__(activation=activation, input_size=input_size, output_size=output_size)

        # matrix of 1 or 0
        # rows = arc from input node
        # cols = arc to output node
        self.weights_mask = np.zeros(self.weights.shape)

        # connections = list of (x, y) where
        # x = row
        # y = col
        for c in connections:
            self.weights_mask[c] = 1

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, np.multiply(self.weights_mask, self.weights)) + self.biases
        return self.activation(self.output)

    def backward(self, output_derivatives, learning_rate):
        undo_activation = np.multiply(output_derivatives, self.activation(self.output, derivative=True))

        input_d = np.dot(undo_activation, np.multiply(self.weights_mask, self.weights).T)  # changed from parent
        d_weights = np.dot(self.input.T, undo_activation)
        d_new_weights = np.multiply(d_weights, self.weights_mask)  # changed from parent

        self.weights = self.weights - d_new_weights * learning_rate
        self.biases = self.biases - undo_activation * learning_rate

        return input_d
