import unittest
from DataLoader import DataLoader
import TrainData
import TestData
import GeneralPurposeRoutine
import CurveAdjusting
from DataLoaderX import DataLoaderX


class TestDataLoader(unittest.TestCase):

    def testgetInstance(self):
        var_xx =TrainData()
        var_xx1 = var_xx.train_model()
        self.assertIsInstance(var_xx1,TrainData(), "Not an instance of this class")




class TestTrainData(unittest.TestCase):
    def test_deviation(self):
        pass


class TestTestData(unittest.TestCase):
    def test_deviation(self):
        pass

class TestGeneralPurposeRoutine(unittest.TestCase):
    def test_deviation(self):
        pass


class TestCurveAdjusting(unittest.TestCase):
    def test_deviation(self):
        pass


if __name__ == '__main__':
    unittest.main()
