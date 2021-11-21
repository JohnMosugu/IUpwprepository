import pandas as pd
import sys
from GeneralPurposeRoutine import GeneralPurposeRoutine
from TestData import TestData
from TrainData import TrainData


class CurveAdjusting(TrainData, TestData):
    """
    This is the main class of the assignment.
    It inherits TestData and TrainData classes and performs the following

    1. Train
    Utlises the training dataset provided to determine the four ideal functions which are the
    best fit out of the fifty provided in ideal dataset. How they minimize
    the sum of all y-deviations squared (Least-Square) is the criteria for
    choosing the ideal functions for the training function.

    2. Test
    Determines for each and every x-y pair of values in test dataset whether
    or not they can be assigned to the four chosen ideal function. The
    requirements to check if or not they can be assigned to the four(4) selected
    ideal function is that the existing maximum deviation of the calculated
    regression for test data does not exceed the greatest deviation between
    training dataset and the ideal function selected for it by more than factor
    square root of 2.

    3. Data Storage in Sqlite database
    Store training data, ideal function data and tested data in sqlite
    database in following format.
    - Training dataset as
            x y1 y2 y3 y4
    - Ideal function dataset as
            x y1 y2 y3 y4 --- y50
    - Tested data as
            x y Mapped_Ideal_Function Deviation

    Parent Class:
        TrainData
        TestData

    Attributes:
        (Class GeneralPurposeRoutine): Database Class Object
    """

    def __init__(self):
        """
        CurveAdjusting class Constructor
        Faciltates initialization of TrainData class, TestData class and Database class.
        It also loads ideal dataset.
        """

        # Initialize TrainData Class
        TrainData.__init__(self)
        # Initialize TestData Class
        TestData.__init__(self)
        # Create Database object to create sqlite
        self.gen_purp_routine = GeneralPurposeRoutine()

    def start(self):
        """
        This method is the entry point of algorithm.
        Here first model is trained and then tested and finally results are
        stored in database.
        """
        # Implementation of Algorithm
        # Train the model
        self.train_model()
        # Test the model
        testresults = self.test_model(self.df_train_results)

        # Create and Update Train, Ideal and Test Tables in sqlite
        self.gen_purp_routine.update_database(testresults)
        # Visualize results
        self.gen_purp_routine.plots(self.df_train_results)


if __name__ == '__main__':
    """if 4 != len(sys.argv):
        print("Usage-----\n curveadjusting.py train_dataset_path "
              "ideal_dataset_path test_dataset_path")
        sys.exit(1)
    """

    try:
        var_run = CurveAdjusting()
    except Exception as e:
        print(str(e))
    else:
        var_run.start()
