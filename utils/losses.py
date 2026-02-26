import math

import numpy as np


class Losses:
    @staticmethod
    def negative_log_likelihood(pred, target_index) -> float:
        pred = [max(min(p, 1 - 1e-15), 1e-15) for p in pred]
        return -math.log(pred[target_index])

    @staticmethod
    def cross_entropy(y_pred, target, eps=1e-9):
        return -np.log(y_pred[target] + eps)

    @staticmethod
    def softmax_ce_grad(y_pred, target):
        dy = y_pred.copy()
        dy[target] -= 1.0
        return dy.reshape(-1, 1)

    @classmethod
    def sequence_nll_with_grads(cls, outputs, targets):
        total_loss = 0.0
        grads = []

        for y_pred, target in zip(outputs, targets):
            y_pred = cls._prepare_prediction(y_pred)

            loss = cls.cross_entropy(y_pred, target)
            grad = cls.softmax_ce_grad(y_pred, target)

            total_loss += loss
            grads.append(grad)

        return total_loss, grads

    @staticmethod
    def _prepare_prediction(y_pred):
        y_pred = np.asarray(y_pred).flatten()
        return y_pred
