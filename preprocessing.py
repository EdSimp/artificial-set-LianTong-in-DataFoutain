# -*- coding:utf-8 -*-
import gc
import math

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def label_change(data):
    lb = LabelEncoder()
    tmp = data.ix[:, ['current_service']].values
    train_label_after_trans = lb.fit_transform(tmp)
    data['current_service'] = train_label_after_trans
    return data


def gender_make(data):
    print('处理性别')
    gender = data['gender'].copy()
    gender[gender == '00'] = 0
    gender[gender == '01'] = 1
    gender[gender == '02'] = 2
    gender[gender == '0'] = 0
    gender[gender == '2'] = 2
    gender[gender == '1'] = 1
    gender = gender[gender != '\\N']  # 可以考虑去掉，只有两条
    data = data[gender != '\\N']
    data['gender'] = gender
    return data


def fill_nan(train, test):
    print("填充Nan值")
    train.replace(to_replace='\\N', value=np.nan, inplace=True)
    test.replace(to_replace='\\N', value=np.nan, inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)


def traffic_level(x):
    if x <= 10000:
        return 0
    elif x > 10000 and x < 20000:
        return 1
    elif x >= 20000:
        return 2
    else:
        return 3


def local_traffic_handle(data):
    # 月累计-本地数据流量
    print("构造local_trafffic_level")
    traffic = data.ix[:, 'local_trafffic_month']
    new = traffic.copy()
    new = new.apply(lambda x: traffic_level(x))
    new_local_traffic = pd.DataFrame(new.values, columns=['local_trafffic_level'])
    res = pd.concat([data, new_local_traffic], axis=1)
    return res


def add_line_four(a, b, c, d):
    return a + b + c + d


def total_fee_average(data):
    # 0.37
    print("构造average_fee")
    tmp = data.apply(lambda row: add_line_four(
        float(row['1_total_fee']), float(row['2_total_fee'])
        , float(row['3_total_fee']), float(row['4_total_fee'])), axis=1)

    data['average_fee'] = tmp / 4

    return data


def min_max_scaler_module(data, name):
    print("将" + name + "的值归一化")
    # age
    data[name] = pd.to_numeric(data[name])
    min_max_scaler = preprocessing.MinMaxScaler()
    tmp = min_max_scaler.fit_transform(data[name].values.reshape(-1, 1))
    data[name] = tmp
    return data


def neg_to_pos(x):
    x = x.astype(float)
    x[x < 0] = 0
    return x


def select_k_feature(data):
    y = data.ix[:, ['current_service']]
    X = data.drop(['current_service', 'user_id'], axis=1)
    X = X.apply(lambda x: neg_to_pos(x))
    X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    return X_new


def mean_pay(data):
    # 0.11
    print("计算mean_pay")
    data['mean_pay'] = data['pay_num'] / data['pay_times']

    return data


def minus_line_two(a, b):
    return a - b


def this_month_real_traffic(data):
    # 0.07
    print("计算这个月真实用trafffic")
    data['real_traffic'] = data['month_traffic'] - data['last_month_traffic']
    return data


def feature_drop(data):
    # data.drop(['last_month_traffic', 'month_traffic'], axis=1)
    data.drop(['former_complaint_fee', 'net_service', 'service1_caller_time'], axis=1)
    return data


def filt_dirty_data(data):
    '''
    交费次数特别多，pay_times
    交费金额特别大，pay_num

    '''
    data = data[data['1_total_fee'] < 5000]
    data = data[data['4_total_fee'] > 0]
    data = data[data['pay_num'] < 50000]
    data = data[data['service2_caller_time'] < 16000]
    return data


def real_traffic(data):
    print("!")


def not_local_traffic(data):
    # 0.04
    data['not_local_traffic'] = data.apply(
        lambda row: minus_line_two(float(row['month_traffic']), float(row['local_trafffic_month'])), axis=1)
    return data


def mean_line_two(a, b):
    if (b != 0):
        a = a / 100
        return a / b
    else:
        return a / 100


def former_ave_fee(data):
    data['former_ave_fee'] = data.apply(
        lambda row: mean_line_two(float(row['former_complaint_fee']), float(row['former_complaint_num'])), axis=1)
    return data


def mutiply_fee(a, b):
    return a ** b


def fee_level_make(data):
    # 没用
    data['fee_level'] = data.apply(
        lambda row: mutiply_fee(float(row['former_complaint_fee']), float(row['complaint_level'])), axis=1)
    data['fee_ave_level'] = data.apply(
        lambda row: mutiply_fee(float(row['former_ave_fee']), float(row['complaint_level'])), axis=1)


def one_hot_feature(data, name):
    print("正在one hot " + name)
    enc = OneHotEncoder()
    new_feat = enc.fit_transform(data.ix[:, name].values.reshape(-1, 1)).toarray()

    namelist = []
    for i in range(new_feat.shape[1]):
        namelist.append(name + '_' + str(i))
    tmp = pd.DataFrame(new_feat, columns=namelist)
    data = data.drop([name], axis=1)
    data = pd.concat([data, tmp], axis=1)
    return data


def total_fee_to_int(combine_data):
    # 2_total_fee
    two_total_fee = combine_data['2_total_fee'].values
    for i in range(len(combine_data['2_total_fee'])):
        if (two_total_fee[i] == '\\N'):
            two_total_fee[i] = float(0)
        else:
            two_total_fee[i] = float(two_total_fee[i])
    combine_data['2_total_fee'] = two_total_fee
    del two_total_fee
    gc.collect()

    # 3_total_fee
    three_total_fee = combine_data['3_total_fee'].values
    for i in range(len(combine_data['3_total_fee'])):
        if (three_total_fee[i] == '\\N'):
            three_total_fee[i] = float(0)
        else:
            three_total_fee[i] = float(three_total_fee[i])
    combine_data['3_total_fee'] = three_total_fee
    del three_total_fee
    gc.collect()

    return combine_data


def add_line_two(a, b):
    return (a + b) / 2


def mean_caller_time(data):
    data['mean_service_time'] = data.apply(lambda row: add_line_two(
        float(row['service1_caller_time']), float(row['service2_caller_time'])), axis=1)
    return data


def classification_row(a):
    if a < 0:
        return -1
    else:
        return 1


def caller_for_free(data):
    print('caller_for_free')
    # fee为负，是送话费
    item_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
    free_list = []
    for item in item_list:
        temp = data.apply(lambda row: classification_row(float(row[item])), axis=1)
        free_list.append(temp)
    free = np.min(free_list, axis=0)
    data['free'] = free
    return data


def type_class(row):
    if (row == '0' or row == '2' or row == '5' or row == '6' or
                row == '7' or row == '8' or row == '9' or row == '10'):
        # 存话费
        return 0
    elif row == '1':
        # 送话费
        return 1
    elif row == '5':
        # 送语音
        return 2
    elif row == '6':
        # 送流量
        return 3
    elif row == '7':
        # 送短信
        return 4
    elif row == '8':
        # 送其他业务
        return 5
    elif row == '11':
        # 送积分
        return 6
    else:
        return 7


def contract_type_class(data):
    print('contract_type_class')
    tmp = data.apply(lambda row: type_class(str(row['contract_type'])), axis=1)
    data['type_class'] = tmp
    return data


def min_total_fee(data):
    print('min_total_free')
    total_fee = pd.DataFrame(data=data, columns=['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'])

    total_fee = np.array(total_fee)
    # total_fee_T=total_fee.T
    min_total_fee = np.min(total_fee, axis=1)
    data['min_total_fee'] = min_total_fee
    data['min_total_fee'] = data['min_total_fee'].apply(lambda x: float(x))
    return data


def real_caller_time(data):
    # 0.29
    print("real caller time")
    data['real_caller_time'] = data['local_caller_time'] - data['service1_caller_time']
    return data


def filter_contract_time(x):
    if (x < 36):
        return x
    else:
        return 40


def add_contract_online_time(data):
    # 0.48
    print('add_contract_online_time')
    data['add_contract_online_time'] = data['online_time'] + data['contract_time']
    data['add_contract_online_time'] = data['add_contract_online_time'].apply(lambda x: filter_contract_time(x))
    return data


def former_mean_money(data):
    # 无用
    print('former_mean_money')
    data['former_mean_money'] = data['former_complaint_fee'] / data['former_complaint_num']
    return data


def traffic_class_function(data):
    if math.ceil(data / 10) < 10:
        class_data = math.ceil(data / 10)
    elif math.ceil(data / 100) < 10:
        class_data = 10 + math.ceil(data / 100)
    elif math.ceil(data / 1000) < 10:
        class_data = 20 + math.ceil(data / 1000)
    elif math.ceil(data / 10000) < 10:
        class_data = 30 + math.ceil(data / 10000)
    elif math.ceil(data / 100000) < 10:
        class_data = 40 + math.ceil(data / 100000)
    elif math.ceil(data / 1000000) < 10:
        class_data = 50 + math.ceil(data / 1000000)
    else:
        class_data = 100
    return class_data


def traffic_class(data):
    # 暂时没用
    '''
    当月流量为0，上个月转结流量为套餐流量
    转结流量为0，当月使用流量上限为套餐流量
    :param data:
    :return:
    '''
    print('traffic_class')
    tmp = data['local_trafffic_month'].apply(lambda x: traffic_class_function(x))
    data['traffic_class'] = tmp
    return data


def groupby_mean_contract_fee(data, name):
    # total_fee_1 0.45 total2 3 4 0.44
    print("groupby_mean_contract_fee")
    contract_fee1_dict = data[name].groupby(data.contract_type).agg('mean').to_dict()
    data['groupby_mean_contract' + name] = data[['contract_type', name]].apply(
        lambda x: x.iloc[1] - contract_fee1_dict[int(x.iloc[0])], axis=1)
    return data


def traffic_fee_func(service_type, month_traffic):
    if service_type == 1:
        return month_traffic / 800
    else:
        return month_traffic * 0.3


def traffic_fee(data):
    # 0.277
    print('traffic_fee')
    data['traffic_fee'] = data.apply(lambda row: traffic_fee_func(row['service_type'], row['month_traffic']), axis=1)
    return data


def traffic_4G_func(service_type, data):
    # 没算结余
    if (service_type == 4):
        # 为4G
        if math.ceil(data) < 500:
            class_data = 1
        elif math.ceil(data) < 800:
            class_data = 2
        elif math.ceil(data) < 1000:
            class_data = 3
        elif math.ceil(data) < 2000:
            class_data = 4
        elif math.ceil(data) < 3000:
            class_data = 5
        elif math.ceil(data) < 4000:
            class_data = 6
        elif math.ceil(data) < 6000:
            class_data = 7
        elif math.ceil(data) < 11000:
            class_data = 8
        else:
            class_data = 8
        return class_data
    else:
        return 0


def traffic_4G_set(data):
    # 0.45
    print('traffic_4G_set')
    data['traffic_4G_set'] = data.apply(lambda row: traffic_4G_func(row['service_type'], row['month_traffic']), axis=1)
    return data


def caller_time_set_func(service_type, caller_time):
    if (service_type == 4):
        if math.ceil(caller_time) < 100:
            return 1
        elif math.ceil(caller_time) < 200:
            return 2
        elif math.ceil(caller_time) < 300:
            return 3
        elif math.ceil(caller_time) < 500:
            return 4
        elif math.ceil(caller_time) < 1000:
            return 5
        elif math.ceil(caller_time) < 2000:
            return 6
        elif math.ceil(caller_time) < 3000:
            return 7
        else:
            return 8
    else:
        return 0


def caller_time_set(data):
    # 0.49
    print('caller_time_set')
    data['caller_time_set'] = data.apply(
        lambda row: caller_time_set_func(row['service_type'], row['local_caller_time']), axis=1)
    return data


def find_min_delta_in_list(x, fee_list):
    x_fee_delta = fee_list[0] - x
    now_fee = fee_list[0]
    for i in range(len(fee_list)):
        now_delta = fee_list[i] - x
        if abs(now_delta) < abs(x_fee_delta):
            x_fee_delta = now_delta
            now_fee = fee_list[i]
    return x_fee_delta, now_fee


def min_delta(x, y, z):
    fee_list = [19, 36, 49, 56, 76, 86, 89, 96, 106, 108, 126, 136, 166, 196, 296]
    # 找到list里最近的数，并找到最小delta
    min_x, x_fee = find_min_delta_in_list(x, fee_list)
    min_y, y_fee = find_min_delta_in_list(y, fee_list)
    min_z, z_fee = find_min_delta_in_list(z, fee_list)
    if abs(min_y) < abs(min_x):
        min_x = min_y
        x_fee = y_fee
    elif abs(min_z) < abs(min_x):
        min_x = min_z
        x_fee = z_fee
    return x_fee


def real_set_fee(data):
    # 19，36,49,56,76,86,89,96,106,108,126,136,166,196,296
    # 找差最小的是套餐
    fee_list = [19, 36, 49, 56, 76, 86, 89, 96, 106, 108, 126, 136, 166, 196, 296]
    print('real_set_fee')
    data['1_fee_set_01'] = data['1_total_fee'] - data['service1_caller_time'] * 0.1
    data['1_fee_set_15'] = data['1_total_fee'] - data['service1_caller_time'] * 0.15
    data['1_fee_set_02'] = data['1_total_fee'] - data['service1_caller_time'] * 0.2
    data['fee_set'] = data.apply(
        lambda row: min_delta(float(row['1_fee_set_01']), float(row['1_fee_set_15']), float(row['1_fee_set_02'])),
        axis=1)
    del data['1_fee_set_01']
    del data['1_fee_set_15']
    del data['1_fee_set_02']
    return data


def midd(kw):
    stack = [i for i in kw if i > 0]
    if len(stack) == 0:
        return 0
    elif len(stack) == 1:
        return stack[0]
    elif len(stack) == 2:
        return np.mean(stack)
    elif len(stack) == 3:
        stack.sort()
        diff = np.abs(np.diff(stack))
        if diff[0] < diff[1]:
            return (stack[0] + stack[1]) / 2
        else:
            return (stack[1] + stack[2]) / 2
    else:
        stack.sort()
        diff = np.abs(np.diff(stack))
        diff_min = np.min(diff)
        if diff_min == diff[0]:
            return (stack[0] + stack[1]) / 2
        elif diff_min == diff[1]:
            return (stack[1] + stack[2]) / 2
        else:
            return (stack[2] + stack[3]) / 2


def skew_fee(data):
    print("skew_data")
    data['fee_skew'] = data[['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']].apply(skew, axis=1)
    return data


def max_fee(data):
    print("max_fee")
    data['fee_max'] = np.max(data[['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']], axis=1)
    return data
