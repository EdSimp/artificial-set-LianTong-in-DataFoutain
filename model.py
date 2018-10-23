# -*- coding:utf-8 -*-
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost.sklearn import XGBClassifier


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'score', f1_score(labels, preds, average='weighted')


def handmade_evalerror(preds, dtrain):
    # tp 正样本判断为正样本
    # fp 负样本判断为正样本
    labels = dtrain.get_label()
    f1_socre_list = 0.0
    for i in range(0, 11):
        tp = preds[(preds == i) & (labels == i)].shape[0]
        fp = preds[(preds == i) & (labels != i)].shape[0]
        fn = preds[(preds != i) & (labels == i)].shape[0]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        f1_socre_list += f1
    res = f1_socre_list / 11
    res = res * res
    return 'score', res


def handmade_evalerror_lgbm(preds, dtrain):
    # tp 正样本判断为正样本
    # fp 负样本判断为正样本
    labels = dtrain.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)

    f1_socre_list = 0.0
    for i in range(0, 11):
        tp = preds[(preds == i) & (labels == i)].shape[0]
        fp = preds[(preds == i) & (labels != i)].shape[0]
        fn = preds[(preds != i) & (labels == i)].shape[0]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        f1_socre_list += f1
    res = f1_socre_list / 11
    res = res * res

    return 'f1_score', res, True


def train_classifier_XGB(x_train, y_train, x_val, y_val):
    print('使用XGB训练：')
    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_val, y_val)
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    param = {'booster': 'gbtree',
             'objective': 'multi:softmax',  # 多分类的问题
             'gamma': 0.05,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
             'max_depth': 7,  # 6  # 构建树的深度，越大越容易过拟合
             'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
             'subsample': 0.8,  # 随机采样训练样本
             'colsample_bytree': 0.8,  # 生成树时进行的列采样
             'min_child_weight': 2,
             # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
             # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
             # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
             'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
             'eta': 0.1,  # 如同学习率 0.2
             'seed': 1000,
             'num_class': 11,
             'early_stopping_rounds': 700
             }
    plist = list(param.items())
    plist += [('eval_metric', 'mlogloss')]
    # plist += [('eval_metric', 'ams@0')]
    num_round = 1500  # 800
    bst = xgb.train(params=plist, dtrain=dtrain, feval=handmade_evalerror, evals=watchlist, num_boost_round=num_round)

    print('训练完毕')
    return bst


def train_classfier_LGBM(x_train, y_train, x_val, y_val, X_test):
    print("使用lightGBM训练")

    lgb_train = lgb.Dataset(x_train.ix[:, 0:].astype('float'), y_train.ix[:, 0].astype('int'))
    lgb_val = lgb.Dataset(x_val.ix[:, 0:].astype('float'), y_val.ix[:, 0].astype('int'))


    params = {
        'learning_rate': 0.035,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 7,
        'num_leaves': 40,  # <2^(max_depth)
        'objective': 'multiclass',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'num_class': 11,
    }

    clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], feval=handmade_evalerror_lgbm,
                    num_boost_round=2000, early_stopping_rounds=200)

    joblib.dump(clf, 'lgb01.pkl')
    # clf = joblib.load('lgb01.pkl')

    xx_pred = clf.predict(X_test, num_iteration=clf.best_iteration)

    xx_pred = [np.argmax(x) for x in xx_pred]

    return clf, xx_pred


def train_sklean_XGB(x_train, y_train, x_val, y_val):
    params = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softmax',
        'num_class': 11,
        # 在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
        'scale_pos_weight': 1,
        'silent': 0
    }
    clf = XGBClassifier(**params)
    grid_params = {
        'learning_rate': np.linspace(0.01, 0.2, 20)  # 得到最佳参数0.01，Accuracy：96.4
    }
    grid = GridSearchCV(clf, grid_params)
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    print("Accuracy:{0:.1f}%".format(100 * grid.best_score_))


def train_classfier_LGBM_kfold(X, Y, X_test, N):
    print("使用lightGBM训练")
    submit = []

    X = np.array(X)
    Y = np.array(Y)
    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

    for k, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print('train _K_ flod', k)
        x_train, x_val, y_train, y_val = X[train_index], X[val_index], Y[train_index], Y[val_index]
        y_train = pd.Series(y_train.reshape(-1))
        y_val = pd.Series(y_val.reshape(-1))

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val)
        params = {
            'learning_rate': 0.05,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            # 'max_depth': 8,
            # 'num_leaves': 55,  # <2^(max_depth)
            'max_depth': 7,
            'num_leaves': 40,  # <2^(max_depth)
            'objective': 'multiclass',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,
            'num_class': 11,
        }

        clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], feval=handmade_evalerror_lgbm,
                        num_boost_round=2000, early_stopping_rounds=200)

        joblib.dump(clf, 'lgb04.pkl')
        # clf = joblib.load('lgb01.pkl')

        xx_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
        xx_pred = [np.argmax(x) for x in xx_pred]

        submit.append(xx_pred)

    submit = np.array(submit)
    submit = submit.T
    res = []

    for i in range(submit.shape[0]):
        res.append(np.argmax(np.bincount(submit[i])))

    return clf, res


def train_classifier_XGB_kfold(X, Y, test_x, N):
    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    print('使用XGB训练：')
    X = np.array(X)
    Y = np.array(Y)
    submit = []

    for k, (train_index, val_index) in enumerate(skf.split(X, Y)):
        x_train, x_val, y_train, y_val = X[train_index], X[val_index], Y[train_index], Y[val_index]

        dtrain = xgb.DMatrix(x_train, y_train)
        dval = xgb.DMatrix(x_val, y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        param = {'booster': 'gbtree',
                 'objective': 'multi:softmax',  # 多分类的问题
                 'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                 'max_depth': 7,  # 6  # 构建树的深度，越大越容易过拟合
                 'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                 'subsample': 0.8,  # 随机采样训练样本
                 'colsample_bytree': 0.8,  # 生成树时进行的列采样
                 'min_child_weight': 2,
                 # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                 # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                 # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                 'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
                 'eta': 0.1,  # 如同学习率 0.2
                 'seed': 1000,
                 'num_class': 11,
                 'early_stopping_rounds': 700
                 }

        plist = list(param.items())
        plist += [('eval_metric', 'mlogloss')]
        # plist += [('eval_metric', 'ams@0')]
        num_round = 2  # 800
        bst = xgb.train(params=plist, dtrain=dtrain, feval=handmade_evalerror, evals=watchlist,
                        num_boost_round=num_round)
        dtest = xgb.DMatrix(test_x)
        prob = bst.predict(dtest)
        submit.append(prob)

    submit = np.array(submit)
    submit = submit.T
    res = []

    for i in range(submit.shape[0]):
        res.append(np.argmax(np.bincount(submit[i].astype(int))))

    print('训练完毕')
    return bst, res
