# -*- coding:utf-8 -*-
import xgboost as xgb
from sklearn.metrics import f1_score

def val_predict_prob(bst, val_x, val_y):
    dval = xgb.DMatrix(val_x)
    prob = bst.predict(dval)
    # res = handmade_evalerror(prob, val_y)
    res = f1_score(y_true=val_y, y_pred=prob, average='weighted')
    print('f1_score is : %s' % res)


def test_predict_prob(bst, test_x):
    print("开始test预测")
    dtest = xgb.DMatrix(test_x)
    prob = bst.predict(dtest)
    return prob

def test_lgb_predict_prob(bst, test_x):
    print("开始test预测")
    prob = bst.predict(test_x)
    return prob
