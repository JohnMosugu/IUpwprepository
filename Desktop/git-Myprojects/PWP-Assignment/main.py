import os
from TestData import TestData
from TrainData import TrainData
from GeneralPurposeRoutine import GeneralPurposeRoutine
import pandas as pd


def main():

    yy = TrainData()
    train_results = yy.train_model()
    xx = TestData()
    print(f'Train Results :{(train_results)}')
    trainResults = pd.DataFrame(train_results)
    print(trainResults)

    test_results = xx.test_model(train_results)
    # print(f'test Result:  {(test_results[1:3])}')

    testresults = pd.DataFrame(data=test_results, columns=['x', 'y', 'Mapped_Ideal_Fn', 'Deviation'])
    print(testresults)
    xx1 = GeneralPurposeRoutine()
    xx1.update_database(test_results)
    # xx1.plots(train_results)


if __name__ == '__main__':
    main()
