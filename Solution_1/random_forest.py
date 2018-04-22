from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# CV
scores = pd.DataFrame(columns=['n_trees', 'performance'])
i = 0
for n in range(20, 161, 20):
    rf = RandomForestClassifier(n_estimators=n, max_features='sqrt')
    score = cross_val_score(rf, X_train, y_train, cv = 10, n_jobs=8, scoring='accuracy').mean()
    scores.loc[i] = [n, score]
    i += 1
print scores

# test set
rf = RandomForestClassifier(n_estimators=20, max_features='sqrt')
rf.fit(X_train, y_train)
pred_y_test = rf.predict(X_test)
print classification_report(y_pred=pred_y_test, y_true=y_test)

# real world
rf = RandomForestClassifier(n_estimators=2, max_features='sqrt')
rf.fit(X, y)
pred_y_test = rf.predict(test)

from write_result import write_result
write_result(test_orig, pred_y_test, 'result/rf.csv')