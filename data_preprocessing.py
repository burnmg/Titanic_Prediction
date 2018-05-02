# from data_exploring import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Handle titles
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]*)\.')

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_survival_means = train_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print train_df.shape, test_df.shape

# Convert Sex
sex_map = {
    'male':0,
    'female':1
}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_map)

# Fill missed ages
## grid = sb.FacetGrid(train_df, row='Pclass', col='Sex')
## grid.map(plt.hist, 'Age')

guess_ages = np.zeros([2,3])
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_ages[i,j] = dataset.loc[(dataset['Sex']==i) & (dataset['Pclass']==j+1),'Age'].median()

for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset['Age'].isna()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j + 1),'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)


# search: pd.cut