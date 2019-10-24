import neural_net as nn
import utils

number_of_neurons_in_input_layer = 2
number_of_neurons_in_hidden_layer = 2
number_of_neurons_in_output_layer = 1

input_layer = nn.Layer()
hidden_layer = nn.Layer()
output_layer = nn.Layer()

input_layer.name = "input layer"
hidden_layer.name = "hidden layer"
output_layer.name = "output layer"

for number in range(number_of_neurons_in_input_layer):
    neuron = nn.Neuron()
    neuron.name = "I{}".format(number)
    neuron.activation_function = utils.unit_activation_function
    input_layer.add_neuron(neuron)

for number in range(number_of_neurons_in_hidden_layer):
    neuron = nn.Neuron()
    neuron.name = "H{}".format(number)
    neuron.activation_function = utils.sigmoid_activation_function
    hidden_layer.add_neuron(neuron)

for number in range(number_of_neurons_in_output_layer):
    neuron = nn.Neuron()
    neuron.name = "O{}".format(number)
    neuron.activation_function = utils.sigmoid_activation_function
    output_layer.add_neuron(neuron)

input_layer.set_output_to_layer(hidden_layer)
hidden_layer.set_output_to_layer(output_layer)

network = nn.Network()
network.add_layer(input_layer)
network.add_layer(hidden_layer)
network.add_layer(output_layer)

network.inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

network.formal_outputs = [
    [0],
    [0],
    [0],
    [1]
]

learning_rate = 0.1
eta = 0.2

network.run(learning_rate=learning_rate, eta=eta)

