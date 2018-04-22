from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import *
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')


'''
importances = pd.DataFrame()
importances['feature'] = train.columns[1:]
importances['importances'] = rf.feature_importances_
importances.sort_values(by=['importances'], ascending=False, inplace=True)
importances.set_index(importances['feature'], inplace=True)
'''


# importances.plot(kind='barh')
score = cross_val_score(rf, X, y, cv = 10, scoring='accuracy')
rf.fit(X=X, y=y)

pred_y_test = rf.predict(test)
from write_result import write_result
write_result(test_orig, pred_y_test, 'data/rf.csv')