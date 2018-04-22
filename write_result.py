import pandas as pd


def write_result(test_orig, pred_y, dir):
    result = pd.concat([test_orig['PassengerId'], pd.DataFrame({'Survived': pred_y})], axis= 1)
    result.to_csv(dir, index=False)