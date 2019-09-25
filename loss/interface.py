from abc import ABCMeta, abstractmethod
import tensorflow as tf

tf.enable_eager_execution()


class Loss(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, y_batch, y_hat_batch, reduction: str = None):
        raise NotImplementedError()

    def __call__(self, y_batch, y_hat_batch, reduction: str = None, *args, **kwargs):
        return self.calculate(y_batch, y_hat_batch, reduction)

    @staticmethod
    def reduce(tensor: tf.Tensor, reduction: str):
        if reduction not in ['min', 'max', 'mean']:
            raise ValueError('reduction can only be one of: min, max or mean.')

        if reduction == 'min':
            return tf.math.reduce_min(tensor)
        if reduction == 'max':
            return tf.math.reduce_max(tensor)
        if reduction == 'mean':
            return tf.math.reduce_mean(tensor)
