def unit_activation_function(value):
    return value


def sigmoid_activation_function(value):
    # lambda equals 0
    return 1 / (1 + 2.718 ** value)


def step_activation_function(value):
    # lambda = 0.5

    if value < 0.5:
        return 0

    return 1


def calculate_error(formal_outputs, actual_outputs):
    error_sum = 0

    for value_number in range(len(formal_outputs)):
        error_sum += (formal_outputs[value_number] - actual_outputs[value_number]) ** 2

    error_sum_root = error_sum ** 1/2

    return error_sum_root/ len(formal_outputs)
