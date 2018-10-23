# -*- coding:utf-8 -*-
import sys

from code_pro.preprocessing import *

sys.path.append('/Users/ayogg/PycharmProjects/JD/code_pro/')

train = pd.read_csv('../data/train_all.csv', sep=',', low_memory=False)
test = pd.read_csv('../data/republish_test.csv', sep=',', low_memory=False)
fill_nan(train, test)
# gender_make(train)
# train = former_ave_fee(train)
# train = total_fee_to_int(train)
# corrmat = train.corr()
# print(corrmat.ix['current_service'].sort_values())
train = caller_time_set(train)
print(train.groupby('caller_time_set')['current_service'].agg({'count'}))

corrmat = train.corr()
print(corrmat.ix['current_service'].sort_values())

tmp = train.groupby(['last_month_traffic', 'local_trafffic_month'],
                    as_index=False).size().reset_index()  # .sort_values(by='current_service')
print(tmp)

data = traffic_class(train)
print(data.groupby('traffic_class')['current_service'].agg({'count'}))
tmp = train['month_traffic'] - train['local_trafffic_month']

# 将label标签化准备分类
lb = LabelEncoder()
tmp = train.ix[:, ['current_ser vice']].values
train_label_after_trans = lb.fit_transform(tmp)
train['current_service'] = train_label_after_trans

tmp = train.groupby(['complaint_level', 'former_complaint_num'], as_index=False).size().reset_index().sort_values(
    by='former_complaint_num')
print(tmp)

corrmat = train.corr()
print(corrmat.ix['current_service'].sort_values())

train.plot(kind='scatter', x='current_service', y='contract_time').get_figure()

# former_ave_fee(train)
# fee_level_make(train)

# train.plot(kind='scatter', x='online_time', y='current_service').get_figure()


tmp.plot(kind='scatter', x='complaint_level', y='online_time').get_figure()

train.plot(kind='scatter', x='current_service', y='service_type').get_figure()
train.plot(kind='scatter', x='current_service', y='service1_caller_time').get_figure()
train.plot(kind='scatter', x='current_service', y='service2_caller_time').get_figure()
# total_fee_average(train)
# train.plot(kind='scatter', x='current_service', y='all_fee').get_figure()

# pay_num = pd.concat([train['service_type'], train['pay_times']], axis=1)

# x = train['current_service']
# y = train['pay_times']

# 散点图

# plt.scatter(x=train['current_service'], y=train['service1_caller_time'])
# plt.ylabel('service1_caller_time', fontsize=13)
# plt.xlabel('current_service', fontsize=13)
'''
for i in range(0, len(train.columns)):
    tmp = train.columns[i]
    if (tmp == 'former_complaint_num'):
        train.plot(kind='scatter', x='current_service', y=train.columns[i]).get_figure()
    if (tmp == 'former_complaint_fee'):
        train.plot(kind='scatter', x='current_service', y=train.columns[i]).get_figure()
    if (tmp == 'age'):
        train.plot(kind='scatter', x='current_service', y=train.columns[i]).get_figure()
    if (tmp == 'gender'):
        train.plot(kind='scatter', x='current_service', y=train.columns[i]).get_figure()
    if (tmp == 'complaint_level'):
        train.plot(kind='scatter', x='current_service', y=train.columns[i]).get_figure()
'''
# plt.show()


# 相关性
'''
大于0.5
contract_time             0.583545
online_time               0.600771
many_over_bill           -0.692560
'''
# corrmat = train.corr()
# print(corrmat.ix['current_service'].sort_values())

# 统计每个元素的个数
'''
local_traffic_month 为0的比较多，其他的是十位数或个位数
gender要合并 00 0 02 01 0 2 1 2 1 #(\  N)

'''

# tmp = train.groupby(['month_traffic'], as_index=False).size().reset_index().sort_values(by=0)
# print(tmp)
# tmp[tmp['month_traffic'] == 0] = 0
# tmp.plot(kind='scatter', x='month_traffic', y=0).get_figure()

# tmp = train.groupby(['current_service', 'last_month_traffic'],as_index = False).size()  # .reset_index().sort_values(by=0)

# print(tmp)
# tmp.to_csv('tmp.csv')


for i in train.columns:
    tmp = train.groupby([i], as_index=False).size().reset_index().sort_values(by=0)
    print(tmp)

print("1")

'''
1. pay_times > 60 ,service_type = 4
'''
'''
online_time可以分level
total_fee 加起来分level
'''

contract_fee1_dict = data['1_total_fee'].groupby(data.contract_type).agg('mean').to_dict()
print(contract_fee1_dict)

groupby_mean_contract_fee1 = data[['contract_type', '1_total_fee']].apply(
    lambda x: x.iloc[1] - contract_fee1_dict[int(x.iloc[0])], axis=1)
print(groupby_mean_contract_fee1)
