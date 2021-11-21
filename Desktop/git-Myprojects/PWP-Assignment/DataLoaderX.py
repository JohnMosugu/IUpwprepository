import os.path
import sys

import pandas as pd


class DataLoaderX:


    def __init__(self):

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
            print("Exception while loading data to dataFrame: ", e)
            sys.exit(1)
        else:
            if self.df_ideal_data.empty:
                print("Empty Ideal Dataset")
                sys.exit(1)
            elif self.df_train_data.empty:
                print("Empty Train Dataset")
                sys.exit(1)
            elif self.df_test_data.empty:
                print("Empty Test Dataset")
                sys.exit(1)
            else:
                print("Data Loading to dataframe successful..")

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
