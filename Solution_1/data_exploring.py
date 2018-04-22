import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

train = pd.read_csv('data/train.csv')
train['Age'] = train['Age'].fillna(train['Age'].median())

sb.barplot(x='Sex', y='Survived', data=train)

sb.violinplot(x='Sex', y='Age', hue='Survived', data=train, split=True)

plt.hist([train[train['Survived'] == 1]['Age'], train[train['Survived'] == 0]['Age']], label = ['Live', 'Dead'])

plt.scatter(x=train['X'], y=train['Fare'])