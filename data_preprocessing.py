# from data_exploring import *
import pandas as pd

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test.df.drop(['Ticket', 'Cabin'], axis=1)

