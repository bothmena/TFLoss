from unittest import TestCase
import numpy as np
from loss.regression import MeanSquaredErrorLoss


class TestMeanSquaredErrorLoss(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MeanSquaredErrorLoss()
        self.y_batch = np.array([[1, 2, 3]], dtype=np.float32)
        self.y_hat_batch = np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=np.float32)

    def test_none_y_batch(self):
        self.assertRaises(ValueError, self.loss, None, np.array([[[1], [2], [3]]]))

    def test_none_y_hat_batch(self):
        self.assertRaises(ValueError, self.loss, np.array([[1, 2, 3]]), None)

    def test_loss_value_reduction_none(self):
        loss_value = self.loss(self.y_batch, self.y_hat_batch)
        self.assertEqual(loss_value.all(), np.array([0, 1, 4], dtype=np.float32).all())
        self.assertEqual(loss_value.shape, (3,))

    def test_loss_value_reduction_min(self):
        loss_value = self.loss(self.y_batch, self.y_hat_batch, reduction='min')
        self.assertEqual(loss_value.all(), np.array(0, dtype=np.float32).all())
        self.assertEqual(loss_value.shape, ())

    def test_loss_value_reduction_max(self):
        loss_value = self.loss(self.y_batch, self.y_hat_batch, reduction='max')
        self.assertEqual(loss_value, np.array(4, dtype=np.float32))
        self.assertEqual(loss_value.shape, ())

    def test_loss_value_reduction_mean(self):
        loss_value = self.loss(self.y_batch, self.y_hat_batch, reduction='mean')
        self.assertEqual(loss_value, np.array(5/3.0, dtype=np.float32))
        self.assertEqual(loss_value.shape, ())
