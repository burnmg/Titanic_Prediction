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


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

ageband_vs_survived = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by = 'AgeBand')
ageband = ageband_vs_survived['AgeBand']

## cut ages into category
dataset.loc[(dataset['Age'] <= ageband[0].right),['Age']] = 0
dataset.loc[(dataset['Age'] > ageband[0].left),['Age']] = len(ageband) - 1
for dataset in combine:
    for i in range(1, len(ageband) - 1):

            dataset.loc[(dataset['Age'] > ageband[i].left) & (dataset['Age'] <= ageband[i].right),['Age']] = i

train_df = train_df.drop(['AgeBand'], axis = 1)
combine = [train_df, test_df]

# Create FamilySize feature
for dataset in combine:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

familysize_vs_survived = train_df[['FamilySize', 'Survived']].groupby(by = ['FamilySize'], as_index = False).mean()
