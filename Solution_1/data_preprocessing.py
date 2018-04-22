import pandas as pd
import matplotlib as mat
import seaborn as sb

mat.interactive(False)

train = pd.read_csv("data/train.csv")

# Check missing values
print train.isnull().sum()
print train.info()

# Handle missing values
train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"], 1) # drop useless variables

# Visualise Age v.s. Pclass
# sb.boxplot(x="Pclass", y="Age", data=train)
# mat.pyplot.show()

# replace missing Age with the mean age of each Pclass
j = (train['Age'].isna()) & (train['Pclass'] == 1)
train.loc[j,'Age'] = train['Age'][train['Pclass']==1].mean()
j = (train['Age'].isna()) & (train['Pclass'] == 2)
train.loc[j,'Age'] = train['Age'][train['Pclass']==2].mean()
j = (train['Age'].isna()) & (train['Pclass'] == 3)
train.loc[j,'Age'] = train['Age'][train['Pclass']==3].mean()

train.info()
## clean rest of missing value
train.dropna(inplace = True)
train.info()

# Create dummy variables
embarked_dummy = pd.get_dummies(train['Embarked'], drop_first=True)
sex_dummy = pd.get_dummies(train['Sex'], drop_first=True)
train.drop(['Embarked', 'Sex'], axis = 1, inplace = True)
train = pd.concat([train, sex_dummy, embarked_dummy],axis = 1)

# check correlation between variables
## sb.heatmap(train.corr(),vmin=-1, vmax=1, cmap='Spectral')

# we found that Fare is correlated with Pclass, we drop one of these
## train.drop(['Fare'], inplace = True, axis = 1)

y = train.iloc[:,0]
X = train.iloc[:,range(1, train.shape[1])]


# processing test data
test_orig = pd.read_csv('data/test.csv')

test = test_orig.drop(["PassengerId", "Name", "Ticket", "Cabin"], 1) # drop useless variables
j = (test['Age'].isna()) & (test['Pclass'] == 1)
test.loc[j,'Age'] = test['Age'][test['Pclass']==1].mean()
j = (test['Age'].isna()) & (test['Pclass'] == 2)
test.loc[j,'Age'] = test['Age'][test['Pclass']==2].mean()
j = (test['Age'].isna()) & (test['Pclass'] == 3)
test.loc[j,'Age'] = test['Age'][test['Pclass']==3].mean()

# Create dummy variables
embarked_dummy = pd.get_dummies(test['Embarked'], drop_first=True)
sex_dummy = pd.get_dummies(test['Sex'], drop_first=True)
test.drop(['Embarked', 'Sex'], axis = 1, inplace = True)
test = pd.concat([test, sex_dummy, embarked_dummy],axis = 1)

# Set Fare for the missing value
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

## test.drop(['Fare'], inplace = True, axis = 1)

