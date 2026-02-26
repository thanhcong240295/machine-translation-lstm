import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x, derivative=False):
        x = np.asarray(x)
        s = 1 / (1 + np.exp(-x))
        if derivative:
            return s * (1 - s)
        return s

    @staticmethod
    def tanh(x, derivative=False):
        x = np.asarray(x)
        t = np.tanh(x)
        if derivative:
            return 1 - t**2
        return t

    @staticmethod
    def softmax(x):
        x = np.asarray(x)

        if x.ndim == 1:
            shifted = x - np.max(x)
            exp_x = np.exp(shifted)
            return exp_x / np.sum(exp_x)

        elif x.ndim == 2:
            shifted = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(shifted)
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)

        else:
            raise ValueError("softmax expects 1D or 2D array")
