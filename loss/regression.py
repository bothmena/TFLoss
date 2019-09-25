from .interface import Loss
import numpy as np


class MeanSquaredErrorLoss(Loss):

    def calculate(self, y_batch: np.ndarray, y_hat_batch: np.ndarray, reduction=None) -> np.ndarray:
        """
        Calculate the loss function using the mean squared error (MSE) formula.

        Parameters
        ----------
        y_batch: np.ndarray
            array of a batch of training targets, a numpy array of shape (M, 3)
        y_hat_batch: np.ndarray
            array of a batch of predictions, a numpy array of shape (M, K, 3)
        reduction: str
            specifies the type of reduction to be used, by default is None, i.e. no reduction will be done and the
            function will return all the computed loss values
            possible values of reduction are: min, max and mean.

        Returns
        -------
        np_loss: np.ndarray
            a numpy array of the (reduced) computed losses
        """
        if y_batch is None or y_hat_batch is None:
            raise ValueError('y_batch and y_hat_batch must not be null.')
        if len(y_batch.shape) != 2:
            raise ValueError('y_batch must be 2-dim array')
        if len(y_hat_batch.shape) != 3:
            raise ValueError('y_hat_batch must be 3-dim array')

        loss = []
        # loop through the M batches
        for i in range(y_hat_batch.shape[0]):
            target = y_batch[i]
            # loop through the K predictions
            for j in range(y_hat_batch.shape[1]):
                y_hat = y_hat_batch[i, j]
                loss_val = 0
                for x, y in zip(target, y_hat):
                    loss_val += ((x - y)**2) / len(target)
                loss.append(loss_val)

        np_loss = np.array(loss, dtype=np.float32).reshape((len(loss), 1))
        if reduction is None:
            return np_loss
        else:
            return self.reduce(np_loss, reduction)
