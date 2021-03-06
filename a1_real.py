#7 删列 改为 换成0


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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectPercentile,GenericUnivariateSelect
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
train1_y=train0_y.apply(float)/800000
'''
# 删有nan的列
train1_x=train1_x.dropna(axis=1,how='any')
#train1_x=train1_x.drop(['Heating','ExterCond','Exterior1st','Exterior2nd','Condition2','RoofMatl'],axis=1)
'''
# 换nan为0
#not_nan=train1_x.isnull().sum()[train1_x.isnull().sum()==0]
have_nan=train1_x.isnull().sum()[train1_x.isnull().sum()!=0]
test1_x[have_nan.index]=test1_x[have_nan.index].fillna(0)
test1_x.fillna(method='ffill',inplace=True)
train1_x.fillna(0,inplace=True)




train_x=train1_x



#train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train1_x,train1_y,test_size=0.4)


##########################################

num_columns=len(train_x.columns)
columns_train_x=train_x.columns
# change type to int
for j in range(num_columns):
    i=columns_train_x[j]
    print('j='+str(j)+',,,i='+i)
    #example=train1_x[i][train1_x[i].notnull()].values[0]
    print(train_x[i].dtype)
    
    
    # jiangwei 
    '''
    if float(sum(train_x[i].value_counts())) / len(train_x[i]) < 0.7:	#不能有太多nan
        print('///////////////**********///////')
        train_x.drop(i,inplace=True,axis=1)
        test_x.drop(i,inplace=True,axis=1)
        test1_x.drop(i,inplace=True,axis=1)
        continue
    
    if float(len(train_x[i].unique())) / len(train_x[i]) > 0.3:		#种类不能太多
        print('KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
        train_x.drop(i,inplace=True,axis=1)
        test_x.drop(i,inplace=True,axis=1)
        test1_x.drop(i,inplace=True,axis=1)
        continue
    
    if float(train_x[i].value_counts().values[0]) / len(train_x[i]) > 0.7:		#不能出现一个寡头
        print('ddddddddddddddddddddddddddddddddd')
        train_x.drop(i,inplace=True,axis=1)
        test_x.drop(i,inplace=True,axis=1)
        test1_x.drop(i,inplace=True,axis=1)
        continue
    '''

    #if type(example) not in [int, float, bool]:
    #if type(example) == str:
    if train_x[i].dtype == 'object':
        print('j='+str(j)+',,,i='+i+',,,1111')
        # encoder start
        lbl=sk.preprocessing.LabelEncoder()
        train_x[i][train_x[i].notnull()]=lbl.fit_transform(train_x[i][train_x[i].notnull()])
        train_x[i] = train_x[i].convert_objects(convert_numeric=True)
        
        #test_x[i][test_x[i].notnull()]=lbl.transform(test_x[i][test_x[i].notnull()])
        #test_x[i] = test_x[i].convert_objects(convert_numeric=True)
        
        test1_x[i][test1_x[i].notnull()]=lbl.transform(test1_x[i][test1_x[i].notnull()])
        test1_x[i] = test1_x[i].convert_objects(convert_numeric=True)

        #example=train1_x[i][train1_x[i].notnull()].values[0]
        #print(type(example))
'''       
sp=SelectPercentile()	#percentile=10
params=[1,2,5,10,15,20,25,30,35,40,50,60,70,80]
grid=GridSearchCV(sp,{'percentile':params})
train_x=grid.fit(train_x,train1_y).transform(train_x)
#test1_x=gus.transform(test1_x) 
'''


gus=GenericUnivariateSelect(param=75)
train_x=gus.fit_transform(train_x,train1_y) 
test1_x=gus.transform(test1_x) 


'''
pca=PCA(n_components=50)
train_x=pca.fit_transform(train_x,train1_y) 
test1_x=pca.transform(test1_x) 
'''

########################

train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train_x,train1_y,test_size=0.0)

########################
dtrain=xgb.DMatrix(train_x,train_y)
#dtest=xgb.DMatrix(test_x)	###

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
            'max_depth':9
            }
num_rounds=1000

#clf=xgb.train(params,dtrain,num_rounds,watchlist)
clf=xgb.train(params,dtrain,num_rounds,watchlist)

'''
#self rate test
dtest_x_self=xgb.DMatrix(test_x)
test_y_pred=pd.Series(clf.predict(dtest_x_self),index=test_y_real.index)
#test_y_pred=test_y_pred*800000

print(math.sqrt(mean_squared_error(test_y_pred,test_y_real)))
'''




# real rate test
dtest1_x_self=xgb.DMatrix(test1_x)
test1_y_pred=pd.Series(clf.predict(dtest1_x_self),index=test0_x.index)*800000
result=pd.DataFrame(test1_y_pred,columns=['SalePrice'])
way_out='/home/m/桌面/House Prices/data/result/'
result.to_csv(way_out+'result.csv')

















