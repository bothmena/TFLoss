from unittest import TestCase
import numpy as np
from loss.regression import MeanSquaredErrorLoss


class TestMeanSquaredErrorLoss(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MeanSquaredErrorLoss()

    def test_none_y_batch(self):
        self.assertRaises(AssertionError, self.loss, None, np.array([[[1], [2], [3]]]))

    def test_none_y_hat_batch(self):
        self.assertRaises(AssertionError, self.loss, np.array([[1, 2, 3]]), None)

    def test_y_batch_
