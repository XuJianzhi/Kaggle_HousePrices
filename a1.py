


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
#import time
#import gc
import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


way_data='/media/m/文档/House Prices/data/'
train0=pd.read_csv(way_data+'train.csv')
train0=train0.set_index('Id')
test0=pd.read_csv(way_data+'test.csv')
test0=test0.set_index('Id')

train0_x=train0.iloc[:,:-1]
train0_y=train0.SalePrice
test0_x=test0

# copy
train1_x=train0_x.copy()
test1_x=test0_x.copy()

# change type to int
for j in range(len(train1_x.columns)):
    i=train1_x.columns[j]
    #print('j='+str(j)+',,,i='+i)
    #example=train1_x[i][train1_x[i].notnull()].values[0]
    print(train1_x[i].dtype)
    #if type(example) not in [int, float, bool]:
    #if type(example) == str:
    if train1_x[i].dtype == 'object':
        print('j='+str(j)+',,,i='+i+',,,1111')
        # encoder start
        lbl=sk.preprocessing.LabelEncoder()
        train1_x[i][train1_x[i].notnull()]=lbl.fit_transform(train1_x[i][train1_x[i].notnull()])
        train1_x[i] = train1_x[i].convert_objects(convert_numeric=True)
        test1_x[i][test1_x[i].notnull()]=lbl.transform(test1_x[i][test1_x[i].notnull()])
        test1_x[i] = test1_x[i].convert_objects(convert_numeric=True)

        #example=train1_x[i][train1_x[i].notnull()].values[0]
        #print(type(example))
    '''
    elif type(example) == np.float64 :
        print('j='+str(j)+',,,i='+i+',,,2222')
        #train1_x[i][train1_x[i].notnull()]=train1_x[i][train1_x[i].notnull()].astype(type('float', (float,), {}))
        train1_x[i]=train1_x[i].astype(type('float', (float,), {}))

        example=train1_x[i][train1_x[i].notnull()].values[0]
        print(type(example))

    elif type(example) == np.int64 :
        print('j='+str(j)+',,,i='+i+',,,3333')
        #xuanze=train1_x[i].notnull()
        #temp=train1_x[i].copy()[xuanze]
        train1_x[i]=train1_x[i].astype(type('int', (int,), {}))
        #train1_x[i][train1_x[i].notnull()]=train1_x[i][train1_x[i].notnull()].astype(float)
        #train1_x[i][train1_x[i].notnull()]=train1_x[i][train1_x[i].notnull()].astype(type('float', (float,), {}))

        example=train1_x[i][train1_x[i].notnull()].values[0]
        print(type(example))

    else :
        print('j='+str(j)+',,,i='+i+',,,4444')
    '''
    
    # jiangwei 
    #if float(sum(train1_x[i].value_counts().values[:3]))/len(train1_x[i]) < 0.5:
    #    train1_x[i].drop(i,inplace=True)



train1_y=train0_y.apply(float)/800000
########################

train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train1_x,train1_y,test_size=0.2)

########################
dtrain=xgb.DMatrix(train_x,train_y)
dtest=xgb.DMatrix(test_x)	###

watchlist=[(dtrain,'train'),(dtrain,'test')]
#num_class=train_y.max()+1  
params = {
            'objective': 'reg:logistic',
            'eta': 0.1,
            'eval_metric': 'rmse',
            #'eval_metric': 'mlogloss',
            'seed': 0,
            'missing': -999,
            #'num_class':num_class,
            'silent' : 1,
            'gamma' : 2,
            'subsample' : 0.5,
            'alpha' : 0.5,
            #'max_depth':2
            }
num_rounds=1000

#clf=xgb.train(params,dtrain,num_rounds,watchlist)
clf=xgb.train(params,dtrain,num_rounds,watchlist)

'''
test_y_pred=pd.Series(clf.predict(dtest),index=test_x.index)
test_y_pred=pd.DataFrame(clf.predict(dtest),index=test_x.index,columns=['Survived'])
test_y['Survived']=test_y['Survived'].apply(int)
test_y=test_y.reset_index()

#way_out='/home/m/Titanic/result/11.20/'
#test_y.to_csv(way_out+'result.csv')
'''


'''
#self all test
dtest_x_self=xgb.DMatrix(train_x)
y_self=pd.Series(clf.predict(dtest_x_self)).apply(int)
y_self=train['Survived']

choice=y_self[y_self==y_self]
right_num=len(choice[choice])	#the result is all right
'''

#self rate test
dtest_x_self=xgb.DMatrix(test_x)
test_y_pred=pd.Series(clf.predict(dtest_x_self),index=test_x.index)
test_y_pred=test_y_pred*800000

print(mean_squared_error(test_y_pred,test_y_real*800000))


'''
choice = y_pred==y_real
right_num=len(choice[choice])	# 0.770949720670391
print(float(right_num)/len(y_pred))
'''
















'''
###############
###############
for i in train1_x.columns:
    print(i+',,,'+str(type(train1_x[i][train1_x[i].notnull()].values[0])))
#
for i in range(a):
    if train_x[j].dtype not in [int,float,bool]:
        print(str(i)+',,,'+str(j)+',,,')
        print(train_x[j].dtype)


'''









