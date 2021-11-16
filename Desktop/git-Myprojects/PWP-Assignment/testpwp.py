import unittest
from DataLoader import DataLoader
import pandas as pd

from TestData import TestData
from TrainData import TrainData


class TestDataShapes(unittest.TestCase):

    def setUp(self):
        var_data = DataLoader()
        self.train_data = var_data.retrieve_train_data()
        self.ideal_data = var_data.retrieve_ideal_data()
        self.test_data = var_data.retrieve_test_data()

    def testCheckTrainDataShape(self):
        returned_shape = pd.DataFrame(self.train_data).shape
        expected_shape = (400, 5)
        self.assertEqual(returned_shape, expected_shape, "Train data is of the wrong dimention/shape")

    def testCheckTestDataShape(self):
        returned_shape = pd.DataFrame(self.test_data).shape
        expected_shape = (100, 2)
        self.assertEqual(returned_shape, expected_shape, "Test data is of the wrong dimention/shape")

    def testCheckIdealDataShape(self):
        returned_shape = pd.DataFrame(self.ideal_data).shape
        expected_shape = (400, 51)
        self.assertEqual(returned_shape, expected_shape, "Ideal data is of the wrong dimention/shape")


class TestTrainTestResults(unittest.TestCase):

    def setUp(self):
        self.var_train = TrainData()
        self.var_train_results = self.var_train.train_model()
        self.var_test = TestData()
        self.var_test_results = self.var_test.test_model(self.var_train_results)

    def test_TestTrainResults(self):
        expected_results= {'y1', 'y2', 'y3', 'y4'}
        return_results = self.var_train_results.keys()
        self.assertEqual(return_results,expected_results,"Train Results is not of the  same shape")

    def test_TestResults(self):
        expected_results = 50
        return_results = len(self.var_test_results)
        self.assertEqual(return_results, expected_results, "Test Results has wrong size")


if __name__ == '__main__':
    unittest.main()
