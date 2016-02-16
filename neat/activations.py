import math
import numpy as np

def sigmoid_activation(x):
    z = np.clip(z,-60.0,60.0)
    return 1.0 / (1.0 + np.exp(-z))


def tanh_activation(x):
    z = np.clip(z,-60.0,60.0)
    return np.tanh(z)


def sin_activation(x):
    z = np.clip(z,-60.0,60.0)
    return np.sin(z)


def gauss_activation(x):
    z = np.clip(z,-60.0,60.0)
    return np.exp(-0.5 * z**2) / np.sqrt(2 * math.pi)


def relu_activation(x):
    return z if z > 0.0 else 0


def identity_activation(x):
    return x


def clamped_activation(x):
    return np.clip(z,-1.0,1.0)


def inv_activation(x):
    if z == 0:
        return 0.0

    return 1.0 / z


def log_activation(x):
    z = np.clip(z,1e-7,1.0e99)
    return np.log(z)


def exp_activation(x):
    z = np.clip(z,-60.0,60.0)
    return np.exp(z)


def abs_activation(x):
    return np.abs(z)


def hat_activation(x):
    return np.clip(z,0.0,1.0-np.abs(z))


def square_activation(x):
    return z ** 2


def cube_activation(x):
    return z ** 3


activations = {'sigmoid':sigmoid_activation,
               'tanh': tanh_activation,
               'sin': sin_activation,
               'gauss': gauss_activation,
               'relu': relu_activation,
               'identity': identity_activation,
               'clamped': clamped_activation,
               'inv': inv_activation,
               'log': log_activation,
               'exp': exp_activation,
               'abs': abs_activation,
               'hat': hat_activation,
               'square': square_activation,
               'cube': cube_activation}


class InvalidActivationFunction(Exception):
    pass


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {}

    def add(self, config_name, function):
        # TODO: Verify that the given function has the correct signature.
        self.functions[config_name] = function

    def get(self, config_name):
        f = self.functions.get(config_name)
        if f is None:
            raise InvalidActivationFunction("No such function: {0!r}".format(config_name))

        return f

    def is_valid(self, config_name):
        return config_name in self.functions


