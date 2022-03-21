import pandas as pd
import sys

class DataLoader:
    """
    This is used to return the instance of the class. In addition, this class will be used to load and share
    data from three csv data files into pandas Dataframe, involving many classes.
    """

    __instance = None
    df_train_data = pd.DataFrame()
    df_ideal_data = pd.DataFrame()
    df_test_data = pd.DataFrame()

    @staticmethod
    def get_instance():
        """
        Static access method to obtain the instance of the class DataLoader.
        :return: DataLoader instance
        """
        if DataLoader.__instance is None:
           DataLoader.__instance= DataLoader()

        return DataLoader.__instance

    def __init__(self):
        """
        The Constructor of the single class DataLoader responsible for loading appropriate datasets
        (train, ideal & test) onto the pandas DataFrames(df_train_data,df_ideal_data & df_test_data)
        """
        try:
            self.df_train_data = pd.read_csv(
                "C:/Users/John Taiye Mosugu/Downloads/Personal stuffs/Data Science Program-IUBH/Python "
                "Programming/Written Assignment-Datasets/train.csv")
            self.df_ideal_data = pd.read_csv(
                "C:/Users/John Taiye Mosugu/Downloads/Personal stuffs/Data Science Program-IUBH/Python "
                "Programming/Written Assignment-Datasets/ideal.csv")
            self.df_test_data = pd.read_csv(
                "C:/Users/John Taiye Mosugu/Downloads/Personal stuffs/Data Science Program-IUBH/Python "
                "Programming/Written Assignment-Datasets/test.csv")
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


