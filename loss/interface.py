from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, y_batch, y_hat_batch):
        raise NotImplementedError()

    def __call__(self, y_batch, y_hat_batch, *args, **kwargs):
        return self.calculate(y_batch, y_hat_batch)
