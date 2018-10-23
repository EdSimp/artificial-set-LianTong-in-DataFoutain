# -*- coding:utf-8 -*-
import warnings

from sklearn.utils import shuffle

from code_pro.model import *
from code_pro.preprocessing import *
from code_pro.val_test import *

warnings.filterwarnings("ignore")


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', f1_score(y_true=labels, y_pred=preds, average='weighted')


def split_train_val(data, ratio):
    print("分离train和val")
    data = shuffle(data)
    num = data.shape[0]
    train = data[0:int(ratio * num)]
    val = data[int(num * ratio):]
    return train, val


def load_xgb_model(name):
    bst = xgb.Booster()
    bst.load_model(name)
    return bst


if __name__ == "__main__":
    print('读取数据')
    sample = pd.read_csv('../data/submit_sample.csv', sep=',')

    train = pd.read_csv('../data/train_all.csv', sep=',')
    test = pd.read_csv('../data/republish_test.csv', sep=',')

    constant_feats = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic', 'pay_times',
                      'pay_num', 'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
                      'service1_caller_time', 'service2_caller_time', 'age', 'former_complaint_num',
                      'former_complaint_fee', 'average_fee', 'mean_service_time', 'free']
    category_feats = ['gender', 'caller_time_set', 'traffic_4G_set', 'service_type', 'contract_type', 'online_time',
                      'contract_time']
    # 离散特征需要one_hot,'contract_time'
    # ,'service_type','online_time','contract_type', 'gender'

    fill_nan(train, test)
    train = gender_make(train)

    # filt_dirty_data(train)

    train = total_fee_to_int(train)
    test = total_fee_to_int(test)

    train[train['2_total_fee'] < 0]['2_total_fee'] = 0
    train[train['3_total_fee'] < 0]['3_total_fee'] = 0
    train[train['4_total_fee'] < 0]['4_total_fee'] = 0

    test[test['2_total_fee'] < 0]['2_total_fee'] = 0
    test[test['3_total_fee'] < 0]['3_total_fee'] = 0
    test[test['4_total_fee'] < 0]['4_total_fee'] = 0

    train = traffic_4G_set(train)
    test = traffic_4G_set(test)

    train = caller_time_set(train)
    test = caller_time_set(test)

    # 离散数据one_hot

    for col in category_feats:
        train = one_hot_feature(train, col)
        test = one_hot_feature(test, col)

    # onehot_train = pd.get_dummies(train[col], prefix=col)
    # train = pd.concat([train, onehot_train], axis=1)
    # onehot_test = pd.get_dummies(test[col], prefix=col)
    # test = pd.concat([test, onehot_test], axis=1)
    # train.drop([col], axis=1, inplace=True)
    # test.drop([col], axis=1, inplace=True)

    train = total_fee_average(train)
    test = total_fee_average(test)

    train = mean_pay(train)
    test = mean_pay(test)

    train = mean_caller_time(train)
    test = mean_caller_time(test)

    train = caller_for_free(train)
    test = caller_for_free(test)

    train = contract_type_class(train)
    test = contract_type_class(test)

    train = min_total_fee(train)
    test = min_total_fee(test)

    train = real_caller_time(train)
    test = real_caller_time(test)

    train = add_contract_online_time(train)
    test = add_contract_online_time(test)

    train = groupby_mean_contract_fee(train, '1_total_fee')
    test = groupby_mean_contract_fee(test, '1_total_fee')

    train = traffic_fee(train)
    test = traffic_fee(test)

    train['2_total_fee'] = pd.to_numeric(train['2_total_fee'])
    train['3_total_fee'] = pd.to_numeric(train['3_total_fee'])
    test['2_total_fee'] = pd.to_numeric(test['2_total_fee'])
    test['3_total_fee'] = pd.to_numeric(test['3_total_fee'])

    # 将性别中变成0123

    # train = feature_drop(train)
    # test = feature_drop(test)

    # 连续数据归一化，需要去掉离群点

    # for col in constant_feats:
    #    train = min_max_scaler_module(train, col)
    #    test = min_max_scaler_module(test, col)

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

    # train = pd.read_csv('train.csv', sep=',')
    # test = pd.read_csv('test.csv', sep=',')

    # 将label标签化准备分类
    lb = LabelEncoder()
    tmp = train.ix[:, ['current_service']].values
    train_label_after_trans = lb.fit_transform(tmp)
    train['current_service'] = train_label_after_trans

    # xgb
    '''
    # 将测试集和训练集分开
    ratio = 0.9
    # 准备train和val和test的数据
    train_data, val_data = split_train_val(train, ratio)
    train_y = train_data.ix[:, ['current_service']]
    train_x = train_data.drop(['current_service', 'user_id'], axis=1)
    val_y = val_data.ix[:, ['current_service']]
    val_x = val_data.drop(['current_service', 'user_id'], axis=1)
    test_x = test.drop(['user_id'], axis=1)
    
    #开始训练
    bst = train_classifier_XGB(train_x.values, train_y.values, val_x.values, val_y.values)
    bst.save_model('../model_save/016.model')
    bst = load_xgb_model('../model_save/015.model')
    
    #开始val和test
    val_predict_prob(bst, val_x.values, val_y.values)
    res = test_predict_prob(bst, test_x.values).astype(int)  
    '''

    # xgb,kfold
    '''
    test_x = test.drop(['user_id'], axis=1)
    train_x = train.drop(['current_service', 'user_id'], axis=1)
    train_y = train.ix[:, ['current_service']]
    
    #训练和预测
    bst, res = train_classifier_XGB_kfold(train_x, train_y, test_x.values, N=2)
    '''

    # lgb

    # 将测试集和训练集分开
    ratio = 0.9
    # 准备train和val和test的数据
    train_data, val_data = split_train_val(train, ratio)
    train_y = train_data.ix[:, ['current_service']]
    train_x = train_data.drop(['current_service', 'user_id'], axis=1)
    val_y = val_data.ix[:, ['current_service']]
    val_x = val_data.drop(['current_service', 'user_id'], axis=1)
    test_x = test.drop(['user_id'], axis=1)

    # 开始训练和预测
    bst, res = train_classfier_LGBM(train_x, train_y, val_x, val_y, test_x)

    # lgb,kfold
    '''
    test_x = test.drop(['user_id'], axis=1)
    train_x = train.drop(['current_service', 'user_id'], axis=1)
    train_y = train.ix[:, ['current_service']]
    #开始test
    bst, res = train_classfier_LGBM_kfold(train_x, train_y, test_x, N=3)  
    '''

    test_label_before_trans = lb.inverse_transform(res)

    # 转换成dataframe
    pred = pd.DataFrame(data=test_label_before_trans, columns=['current_service'])
    sample['current_service'] = pred

    # 保存csv
    sample.to_csv('submit_xgb_1.csv', index=False, encoding='gbk')
    # sample.to_csv('submit_xgb.csv', index=False, encoding='gbk')
