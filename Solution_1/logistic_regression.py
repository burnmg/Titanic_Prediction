from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_preprocessing import *




X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=.3)
logis = LogisticRegression()
logis.fit(X_train, y_train)

y_pred = logis.predict(X_validation)

y_pred_test = logis.predict(test)

print classification_report(y_validation, y_pred)

