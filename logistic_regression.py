from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_preprocessing import *


y = train.iloc[:,0]
X = train.iloc[:,range(1, train.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
logis = LogisticRegression()
logis.fit(X_train, y_train)

y_pred = logis.predict(X_test)

print classification_report(y_test, y_pred)
