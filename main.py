import xml.etree.ElementTree as ET

from neural_network import NeuralNetwork


def select_gate():
    print("Select the logical operation to train:")
    print("1. AND")
    print("2. OR")
    print("3. NAND")
    print("4. NOR")
    option = int(input("Enter the option number: "))
    return option

def configure_inputs(option):
    if option == 1:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 0, 0, 1], "AND"
    elif option == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 1], "OR"
    elif option == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 1, 0], "NAND"
    elif option == 4:
        return [(0, 0), (0, 1), (1, 0), (1, 1)], [1, 0, 0, 0], "NOR"
    else:
        print("Invalid option.")
        return None, None, None

def save_weights_to_xml(network, xml_file):
    root = ET.Element("neural_network")
    for idx, layer in enumerate(network.layers):
        layer_element = ET.SubElement(root, f"layer_{idx}")
        for j, neuron in enumerate(layer):
            neuron_element = ET.SubElement(layer_element, f"neuron_{j}")
            ET.SubElement(neuron_element, "weights").text = ",".join(map(str, neuron.weights))
            ET.SubElement(neuron_element, "bias").text = str(neuron.bias)
    tree = ET.ElementTree(root)
    tree.write(xml_file)
    print(f"\nWeights and bias saved to {xml_file}")


def main():
    option = select_gate()
    inputs, expected_results, gate_name = configure_inputs(option)
    if not inputs:
        return

    # Change the input to add the last output layer
    neurons_per_hidden_layer = input("Enter the number of neurons per hidden layer (separated by commas): ")

    # Process the input and ensure the last layer is always 1
    structure = [2]  # Input layer with 2 neurons
    if neurons_per_hidden_layer:
        hidden_layers = [int(n) for n in neurons_per_hidden_layer.split(",")]
        structure.extend(hidden_layers)
    structure.append(1)  # Output layer always with 1 neuron

    network = NeuralNetwork(structure)

    learning_rate = 0.1
    cycles = 50000

    # Train the network
    network.train(inputs, expected_results, learning_rate, cycles)

    # Save the weights to an XML file
    xml_file = f"weights_{gate_name}.xml"
    save_weights_to_xml(network, xml_file)

    # Test the trained network
    print(f"\nTest of the trained network for {gate_name}:")
    for input_data in inputs:
        result = network.forward_pass(input_data)
        print(f"Input: {input_data}, Output: {result[0]:.5f}")


if __name__ == "__main__":
    main()
