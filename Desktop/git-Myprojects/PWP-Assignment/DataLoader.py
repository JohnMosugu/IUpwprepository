import sys

import pandas as pd


class DataLoader:
    """
    Singleton class implementation required to provide global access to our dataset files

    This class will be used loaded and share data from data files into pandas
    Dataframe, involving mainly 3 classes.

    Attributes:
        df_train_data (pandas.core.frame.DataFrame): Training dataset
        df_ideal_data (pandas.core.frame.DataFrame): Ideal dataset
        df_test_data (pandas.core.frame.DataFrame): Test dataset

    """
    __instance = None
    df_train_data = pd.DataFrame()
    df_ideal_data = pd.DataFrame()
    df_test_data = pd.DataFrame()

    @staticmethod
    def get_instance():
        """
        Static access method to obtain the instance of the singleton class DataLoader.
        :return: DataLoader instance
        """
        return DataLoader() if DataLoader.__instance is None else \
            DataLoader.__instance

    def __init__(self):
        """
        The Constructor of the single class DataLoader responsible for loading appropriate datasets
        (train, ideal & test) onto the pandas DataFrames(df_train_data,df_ideal_data & df_test_data)
        """
        # Ensuring it is a virtually private constructor.
        if DataLoader.__instance is not None:
            raise Exception("Only singleton class allowed here!")
        else:
            DataLoader.__instance = self
            try:
                self.df_train_data = pd.read_csv(sys.argv[1])
                self.df_ideal_data = pd.read_csv(sys.argv[2])
                self.df_test_data = pd.read_csv(sys.argv[3])
            except Exception as e:
                print("Exception while loading dataframe :", e)
                sys.exit(1)
            else:
                if self.df_train_data.empty:
                    print("Empty training dataset encountered")
                    sys.exit(1)
                elif self.df_ideal_data.empty:
                    print("Empty Ideal dataset encountered")
                    sys.exit(1)
                elif self.df_test_data.empty:
                    print("Empty Test dataset encountered")
                    sys.exit(1)
                else:
                    print("Data loaded to dataframe(s) successfully...")

    def retrieve_train_data(self):
        """
        Used to get Train dataset
        :return: pandas.core.frame.DataFrame
        """
        return self.df_train_data

    def retrieve_ideal_data(self):
        """
        Used to get Ideal dataset
        :return: pandas.core.frame.DataFrame
        """
        return self.df_ideal_data

    def retrieve_test_data(self):
        """
        Used to get Test dataset
        :return: pandas.core.frame.DataFrame
        """
        return self.df_test_data


# Create the instance of DataLoader
z = DataLoader()