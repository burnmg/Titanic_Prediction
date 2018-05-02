from libraries_list import *


train_df = pd.read_csv('data/train.csv')

# Check overview of the data
train_df.info()
train_df.describe()
train_df.describe(include=[np.object]) # Check uniqueness of each feature

# Check missing value
train_df.isna().sum()

# Check correlation
# sb.heatmap(train_df.corr(), vmin = -1, vmax= 1, cmap='RdBu_r')

# Pivoting Analysis
print train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
print train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
print train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Visualization Analysis
## Does age affect Survival

plt.hist([train_df['Age'][(np.isfinite(train_df['Age'])) & (train_df['Survived'] == 1)]
          ,train_df['Age'][(np.isfinite(train_df['Age'])) & (train_df['Survived'] == 0)]
          ], histtype='bar', stacked = True, label=['Survived', 'Dead'], color = ['g', 'r'])
train_df[['Survived', 'Age']].plot(kind='hist', stacked=True)
plt.show()

grid = sb.FacetGrid(train_df, col='Sex', row='Survived')
grid.map(plt.hist, 'Age')

pclass_survived = pd.DataFrame(
    {'Survived':train_df['Survived'].values,
    'Dead': abs(train_df['Survived'].values - 1),
    'Pclass': train_df['Pclass']})

pclass_survived = pclass_survived.groupby(by='Pclass').sum()
pclass_survived.plot(kind='bar', stacked = True)