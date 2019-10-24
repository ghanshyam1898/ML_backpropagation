import numpy
import random
import utils


class Neuron:
    def __init__(self):
        self.name = "anonymous neuron"
        self.weights = []
        self.activation_function = utils.unit_activation_function
        self.input_value = None
        self.d_value = None
        self.last_change_in_weights = []

    def get_activated_output(self):
        return self.activation_function(self.input_value)

    def __str__(self):
        return self.name


class Layer:
    def __init__(self):
        self.name = "anonymous layer"
        self.neurons = []
        self._output_to_layer = None
        self.input_from_layer = None

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def set_output_to_layer(self, layer):
        self._output_to_layer = layer
        layer.input_from_layer = self

        for neuron in self.neurons:
            neuron.weights = [random.random() for _ in layer]  # initialise weights to random values
            neuron.last_change_in_weights = [0 for _ in layer]  # fill zeroes

    def get_output_to_layer(self):
        return self._output_to_layer

    def size(self):
        return len(self.neurons)

    def __getitem__(self, item):
        return self.neurons[item]

    def __setitem__(self, key, value):
        self.neurons[key] = value

    def __str__(self):
        return self.name


class Network:
    def __init__(self):
        self.layers = []
        self.inputs = []
        self.formal_outputs = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def run(self, learning_rate, eta):
        for input_set_number in range(len(self.inputs)):
            for i in range(self.layers[0].size()):
                self.layers[0][i].input_value = self.inputs[input_set_number][i]

            for layer in self.layers:
                if layer.get_output_to_layer() is not None:
                    for output_neuron_number in range(layer.get_output_to_layer().size()):
                        output_value = 0

                        for input_neuron_number in range(layer.size()):
                            output_value += layer[input_neuron_number].get_activated_output() \
                                            * layer[input_neuron_number].weights[output_neuron_number]

                        layer.get_output_to_layer()[output_neuron_number].input_value = output_value

            output_layer = self.layers[-1]
            error = utils.calculate_error(self.formal_outputs[input_set_number],
                                          [neuron.get_activated_output() for neuron in output_layer])

            current_layer = output_layer
            previous_layer = current_layer.input_from_layer

            # calculate and set the value "d" for each neuron
            # vector d is calculated for final o/p layer only
            for output_neuron_number in range(output_layer.size()):
                formal_output = self.formal_outputs[input_set_number][output_neuron_number]
                actual_output = output_layer[output_neuron_number].get_activated_output()

                output_layer[output_neuron_number].d_value = (formal_output - actual_output) * actual_output * (1 - actual_output)

            d_vector = numpy.array([[neuron.d_value for neuron in current_layer]])

            # calculate matrix Y
            output_of_previous_layer = numpy.array([[neuron.get_activated_output() for neuron in previous_layer]])
            output_of_previous_layer = output_of_previous_layer.T
            matrix_y = numpy.matmul(output_of_previous_layer, d_vector)

            # calculate delta w
            old_delta_w = numpy.array([neuron.last_change_in_weights for neuron in previous_layer])
            new_delta_w = (learning_rate * old_delta_w) + (eta * matrix_y)

            # calculate vector e
            matrix_w = [neuron.weights for neuron in previous_layer]
            vector_e = numpy.matmul(matrix_w, d_vector)

            # calculate d*
            d_star = []
            for neuron_number in range(previous_layer.size()):
                OHi = previous_layer[neuron_number].get_activated_output()
                d_star_value = (vector_e[neuron_number] * OHi )*(1 - OHi)
                d_star.append(d_star_value)

            # update the weights by adding del w and set new delta_w as old delta_w
            for neuron_number in range(previous_layer.size()):
                new_weights = []
                old_weights = previous_layer[neuron_number].weights
                previous_layer.last_change_in_weights = new_delta_w[neuron_number]

                for weight_number in range(len(old_weights)):
                    new_weights.append(old_weights[weight_number] + new_delta_w[neuron_number][weight_number])

                previous_layer[neuron_number].weights = new_weights

            current_layer = previous_layer  # propagate back
            previous_layer = previous_layer.input_from_layer

            while previous_layer is not None:

                # calculate vector e
                matrix_w = [neuron.weights for neuron in previous_layer]
                vector_e = numpy.matmul(matrix_w, d_vector)

                # calculate d*
                d_star = []
                for neuron_number in range(previous_layer.size()):
                    OHi = previous_layer[neuron_number].get_activated_output()
                    d_star_value = (vector_e[neuron_number] * OHi )*(1 - OHi)
                    d_star.append(d_star_value)

                # update the weights by adding del w and set new delta_w as old delta_w
                for neuron_number in range(previous_layer.size()):
                    new_weights = []
                    old_weights = previous_layer[neuron_number].weights
                    previous_layer.last_change_in_weights = new_delta_w[neuron_number]

                    for weight_number in range(len(old_weights)):
                        new_weights.append(old_weights[weight_number] + new_delta_w[neuron_number][weight_number])

                    previous_layer[neuron_number].weights = new_weights

                current_layer = previous_layer  # propagate back
                previous_layer = previous_layer.input_from_layer

            print("------------------------ Iteration {} ------------------------".format(input_set_number))
            print("Error : {:.2f}".format(error))
            self.print_network()
            print("--------------------------------------------------------------\n\n")

    def print_network(self):
        for layer in self.layers:
            print("Layer name : {}".format(layer.name))
            print("Neuron name        input value       output value   weights")
            for neuron in layer:
                print("{}                 {:.2f}              {:.2f}           {}".format(neuron.name,
                                                                                          neuron.input_value,
                                                                                          neuron.get_activated_output(),
                                                                                          ["{:.2f}".format(x) for x
                                                                                           in neuron.weights]))
            print("")
