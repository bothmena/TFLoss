from abc import ABCMeta, abstractmethod
import numpy as np


class Loss(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, y_batch, y_hat_batch, reduction: str = None):
        raise NotImplementedError()

    def __call__(self, y_batch, y_hat_batch, reduction: str = None, *args, **kwargs):
        return self.calculate(y_batch, y_hat_batch, reduction)

    @staticmethod
    def reduce(array: np.array, reduction: str):
        if reduction not in ['min', 'max', 'mean']:
            raise ValueError('reduction can only be one of: min, max or mean.')

        if reduction == 'min':
            return np.min(array)
        if reduction == 'max':
            return np.max(array)
        if reduction == 'mean':
            return np.mean(array)
