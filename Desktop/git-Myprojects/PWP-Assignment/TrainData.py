import numpy as np
from DataLoader import DataLoader


class TrainData:
    """
    Model Training class
    This class utilises the training dataset to determine the four ideal functions
    which are the best fit out of the fifty provided in ideal dataset.
    The approach/criteria  adopted is how they minimize the sum of all y-deviations squared (Least-Square)
    for choosing the ideal functions for the training function.

    Attributes:
        df_train (pandas.core.frame.DataFrame): Training dataset
        df_ideal (pandas.core.frame.DataFrame): Ideal dataset
        df_train_results (dict): Stores 4 chosen ideal functions together with the
        Maximum deviation, minimum sum of square deviations for each training
        function.
        {'training_function': ('Ideal_function',  'Maximum_deviation',
        'minimum_sum_squared_deviations')}
    """

    def __init__(self):
        """
        Constructor of TrainModel class
        """
        # Load Train dataset
        dl = DataLoader.get_instance()
        self.df_train_data = dl.retrieve_train_data()
        self.df_ideal_data = dl.retrieve_ideal_data()
        self.df_train_results = {}

    def train_model(self):
        """
        Method used to map each training function to ideal function
        based on minimum sum of squares deviation criteria. And it also stores
        the results in a dictionary object with following structure
        {'training_function': ('Ideal_function', 'Maximum_deviation',
                                'minimum_sum_squared_deviations')}
        """
        # For each column in the train dataset check for a best fit
        for t_col in self.df_train_data.columns[1:]:
            # Set local parameters
            least_square_error = sys.maxsize
            # For each ideal function check
            # if best fit for current train column
            for i_col in self.df_ideal_data.columns[1:]:
                # Calculate deviation
                dev = np.absolute(np.subtract(self.df_train_data[
                                                  t_col].to_numpy(),
                                              self.df_ideal_data[i_col].to_numpy()))

                # Calculate sum of squared deviations
                lse = np.sum(np.square(dev))
                # Check if current sum of squared deviations minimized
                # if true update the local parameters
                if lse < least_square_error:
                    least_square_error = lse
                    # Keep/Store best fit ideal function
                    # and related parameters in results
                    self.df_train_results[t_col] = (i_col, dev.max(),
                                                    least_square_error)

        return self.df_train_results
