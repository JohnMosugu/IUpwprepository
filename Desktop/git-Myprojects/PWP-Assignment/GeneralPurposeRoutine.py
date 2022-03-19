import sqlalchemy as db
from bokeh.plotting import figure, output_notebook, output_file, save, show
from DataLoader import DataLoader
from TestData import TestData
import pandas as pd
from TrainData import TrainData


class GeneralPurposeRoutine:
    """
    This class handles data storage and visualization

    1. It handles creation of sqlite db file and in it creates 3 Tables shown below
    Train, Ideal and Test.
    The four columns of Train Table are
        x y1 y2 y3 y4
    Those of the Ideal Table are
        x y1 y2 y3 y4 y5 y6--- y50
    Finally the Test Table looks like
        x y Mapped Ideal Function Deviation

    2. This class is also used to plot scatter plots of train dataset together with best
    fit ideal function

    Attributes:
    engine (sqlalchemy.engine.base.Engine): Engine Object Reference
    df_ideal (pandas.core.frame.DataFrame): Ideal dataset
    df_train (pandas.core.frame.DataFrame): Train dataset
    """

    def __init__(self):
        """
        Database class constructor. Sqlite Database created as part of initialisation
        """
        # Create the sqlite database
        self.engine = db.create_engine(f'sqlite:///pwp-assignment.db')

        dl = DataLoader.get_instance()
        self.df_train = dl.retrieve_train_data()
        self.df_ideal = dl.retrieve_ideal_data()

    @staticmethod
    def filename(pre):
        return f'{pre}.html'

    def update_database(self, testresults):
        """
        This method is used to create and update Tables in the sqlite database
        setup during class initialization.
        Train, Ideal and Test Tables

        :param df_test_results:(pandas.core.frame.DataFrame): Data for Test Table

        """
        df_test_results = pd.DataFrame(testresults, columns=['x', 'y', ' mapped_ideal_fn', ' Deviation'])
        # print(df_test_results.head())

        # Create and update Train Table in sqlite database
        self.df_train.to_sql('Train', con=self.engine, if_exists='replace', index=False)

        # Create and update Ideal Table in sqlite database
        self.df_ideal.to_sql('Ideal', con=self.engine, if_exists='replace', index=False)

        # Create and update Test results Table in sqlite database
        df_test_results.to_sql('Test', con=self.engine, if_exists='replace', index=False)

    def plots(self, df_train_results):
        """
        This method is used to plot scatter plot for train data and chosen ideal data on single plot so as to check if they fit.

        :param df_train_results:(dict): Only four Best fit results
       """
        self.df_train_results = df_train_results

        # For each function in Train dataset
        for col in self.df_train.columns[1:]:
            # Return chosen function mapped to the latest train function
            ideal_fn = df_train_results[col][0]
            # Create a scatter plot
            create_plt = figure(title=f'Ideal Function {ideal_fn} Vs  Train 'f'Function {col}', x_axis_label=f'Train '
                                                                                                             f'Function{col}',
                                y_axis_label=f'Ideal Function {ideal_fn}')
            create_plt.scatter(self.df_ideal['x'], self.df_ideal[ideal_fn], size=5, color='red', alpha=0.5)

            create_plt.scatter(self.df_train['x'], self.df_train[col], size=3, color='blue', alpha=.8)
            # Display the mapped chosen ideal functions to training functions
            print(f'Mapped Chosen Ideal Function: {ideal_fn}')
            print(f'Train Result : {col}')

            output_file(f'Output_functions{ideal_fn + col}.html',
                        title=f'Ideal Function-{ideal_fn} Vs  Train 'f'Function-{col}')
            save(create_plt)
            show(create_plt)
