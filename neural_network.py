from neuron import Neuron


class NeuralNetwork:
    def __init__(self, structure):
        # Structure is a list that indicates how many neurons each layer has
        self.layers = []
        for i in range(1, len(structure)):
            layer = [Neuron(structure[i - 1]) for _ in range(structure[i])]
            self.layers.append(layer)

    def forward_pass(self, inputs):
        activations = inputs
        for layer in self.layers:
            new_activation = [neuron.output(activations) for neuron in layer]
            activations = new_activation
        return activations

    def backpropagation(self, inputs, expected_result, learning_rate):
        # Step 1: Forward pass
        activations = [inputs]
        for layer in self.layers:
            layer_activation = [neuron.output(activations[-1]) for neuron in layer]
            activations.append(layer_activation)

        # Display activations for debugging
        print(f"Activations: {activations}")

        # Step 2: Calculate errors in the output layer
        errors = [None] * len(self.layers)
        errors[-1] = [
            (expected_result[i] - activations[-1][i]) * self.layers[-1][i].sigmoid_derivative(activations[-1][i])
            for i in range(len(activations[-1]))
        ]

        # Display output errors for debugging
        print(f"Output errors: {errors[-1]}")

        # Step 3: Backpropagate errors
        for i in range(len(self.layers) - 2, -1, -1):
            errors[i] = [
                sum(errors[i + 1][j] * self.layers[i + 1][j].weights[k] for j in range(len(self.layers[i + 1]))) *
                self.layers[i][k].sigmoid_derivative(activations[i + 1][k])
                for k in range(len(self.layers[i]))
            ]

            # Display hidden layer errors for debugging
            print(f"Errors in layer {i}: {errors[i]}")

        # Step 4: Update weights and biases
        for i in range(len(self.layers)):
            for j, neuron in enumerate(self.layers[i]):
                neuron.adjust_weights(activations[i], errors[i][j], learning_rate)

    def train(self, inputs, expected_results, learning_rate, cycles):
        for cycle in range(cycles):
            for input_data, result in zip(inputs, expected_results):
                self.backpropagation(input_data, [result], learning_rate)
