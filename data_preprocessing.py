import pandas as pd
import matplotlib as mat

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

