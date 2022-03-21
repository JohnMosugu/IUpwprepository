import numpy as np
from DataLoader import DataLoader
from TrainData import TrainData


class TestData:
    """
    Method used to map test data to ideal functions chosen during training.
    Determines for all x-y pairs of values in the test dataset whether
    or not they can be assigned to the four chosen ideal function.
    The dependencies to check whether or not they can be assigned to the four chosen
    ideal function is that the existing maximum deviation of the calculated
    regression for test data does not exceed the largest deviation between
    training dataset and the ideal function chosen for it by more than factor
    sqrt(2).

    Attributes:
        df_test_data (pandas.core.frame.DataFrame): Test dataset
        df_ideal_data (pandas.core.frame.DataFrame): Ideal dataset
        df_test_results (list): Test results as
            [(x, y, mapped_ideal_function, deviation)]
    """

    def __init__(self):
        """
        TestData class constructor
        """
        # self.df_train_results = df_train_results
        # Load Test dataset
        dl = DataLoader.get_instance()
        self.df_test_data = dl.retrieve_test_data()
        self.df_ideal_data = dl.retrieve_ideal_data()
        self.df_test_results = []

    def test_model(self, df_train_results):

        """
         This function maps every x-y pair in test dataset to an
        ideal function chosen while training the model. The preconditions to check
        whether or not they can be assigned to the four chosen ideal function
        is that the existing maximum deviation of the calculated regression
        for test data is not greater than the largest deviation between training
        dataset and the ideal function chosen for it by more than a factor sqrt(2)
        And it also Store test results in following template
        [(x, y, mapped_ideal_function, deviation)]

        Parameters:
            df_train_results (dict): Training result stored in following format
            {'training_function': ('Ideal_function', 'Maximum_deviation',
            'minimum_sum_squared_deviations')}
        """
        # initialise train results
        self.df_train_results = df_train_results

        # looping through every x-y pair in the Test dataset
        for i in range(self.df_test_data.shape[0]):
            # For every row in the train results
            for k in self.df_train_results.keys():
                # Return the chosen ideal function
                ideal_fn = self.df_train_results[k][0]
                # Maximum deviation obtained from the train dataset relating to ideal function chosen
                max_dev = self.df_train_results[k][1]
                # Determine 'x' value in the ideal dataset
                index = np.where(self.df_ideal_data['x'] == self.df_test_data.iloc[i][0])
                # Compute the deviation of test 'y' from the ideal function chosen for 'x'
                dev = np.absolute(np.subtract(self.df_test_data.iloc[i][1],
                                              self.df_ideal_data[ideal_fn].iloc[index].to_numpy()))
                # Cross-check that deviation obtained is less than the product of the square root of 2 and the
                # maximum deviation
                if np.sqrt(2) * max_dev > dev[0]:
                    # Just store the value in the Test results by simply appending it
                    # Break and then return the results required for updating the Test table in Sqlite database
                    self.df_test_results.append((self.df_test_data.iloc[i][0],
                                                 self.df_test_data.iloc[i][1],
                                                 ideal_fn, dev[0]))
                    break
        return self.df_test_results
