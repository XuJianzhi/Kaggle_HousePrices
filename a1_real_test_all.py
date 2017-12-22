


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
import math

way_data='/media/m/文档/House Prices/data/'
train0=pd.read_csv(way_data+'train.csv')
train0=train0.set_index('Id')
test0=pd.read_csv(way_data+'test.csv')
test0=test0.set_index('Id')

train0_x=train0.iloc[:,:-1]
train0_y=train0.SalePrice
test0_x=test0



##########################################
# copy
train1_x=train0_x.copy()
test1_x=test0_x.copy()

num_columns=len(train1_x.columns)
columns_train1_x=train1_x.columns
# change type to int
for j in range(num_columns):
    i=columns_train1_x[j]
    print('j='+str(j)+',,,i='+i)
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
    # jiangwei 
    if float(sum(train1_x[i].value_counts().values[:5]))/len(train1_x[i]) < 0.5:
        print('///////////////**********///////')
        train1_x.drop(i,inplace=True,axis=1)
        test1_x.drop(i,inplace=True,axis=1)
    '''
train1_y=train0_y.apply(float)/800000
########################

#train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train1_x,train1_y,test_size=0.5)

train_x, train_y = train1_x, train1_y
test_x, test_y_real = train1_x, train1_y
########################
dtrain=xgb.DMatrix(train_x,train_y)
dtest=xgb.DMatrix(test_x)	###

watchlist=[(dtrain,'train'),(dtrain,'test')]
#num_class=train_y.max()+1  
params = {
            'objective': 'reg:gamma',
            'eta': 0.01,
            'eval_metric': 'rmse',
            #'eval_metric': 'mlogloss',
            'seed': 0,
            'missing': -999,
            #'num_class':num_class,
            'silent' : 1,
            'gamma' : 1,
            'subsample' : 0.5,
            'alpha' : 0.5,
            'max_depth':5
            }
num_rounds=500

#clf=xgb.train(params,dtrain,num_rounds,watchlist)
clf=xgb.train(params,dtrain,num_rounds,watchlist)


#self rate test
dtest_x_self=xgb.DMatrix(test_x)
test_y_pred=pd.Series(clf.predict(dtest_x_self),index=test_x.index)
test_y_pred=test_y_pred#*800000

print(math.sqrt(mean_squared_error(test_y_pred,test_y_real)))





# real rate test
dtest1_x_self=xgb.DMatrix(test1_x)
test1_y_pred=pd.Series(clf.predict(dtest1_x_self),index=test1_x.index)*800000
result=pd.DataFrame(test1_y_pred,columns=['SalePrice'])
way_out='/media/m/文档/House Prices/data/result/'
result.to_csv(way_out+'result.csv')


















