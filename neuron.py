import math
import random


class Neuron:
    def __init__(self, num_inputs):
        # Initializes the weights and bias randomly for each neuron
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = random.random()

    def output(self, inputs):
        # Calculates the neuron's output using the sigmoid activation function
        sum_output = sum(p * e for p, e in zip(self.weights, inputs)) + self.bias
        return self.sigmoid_function(sum_output)

    def sigmoid_function(self, output):
        return 1 / (1 + math.exp(-output))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def adjust_weights(self, inputs, error, learning_rate):
        # Adjusts the weights and bias using the delta rule
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * error * inputs[i]
        self.bias += learning_rate * error
