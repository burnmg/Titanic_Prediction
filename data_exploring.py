import pandas as pd
import seaborn as sb

train = pd.read_csv('data/train.csv')
train['Age'] = train['Age'].fillna(train['Age'].median())

sb.barplot(x='Sex', y='Survived', data=train)

sb.violinplot(x='Sex', y='Age', hue='Survived', data=train, split=True)