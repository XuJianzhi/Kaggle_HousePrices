


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
import math
#import time
#import gc
import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,GenericUnivariateSelect
from sklearn.preprocessing import Imputer,OneHotEncoder,PolynomialFeatures


way_data='/home/m/桌面/House Prices/data/'
train0=pd.read_csv(way_data+'train.csv')
train0=train0.set_index('Id')
test0=pd.read_csv(way_data+'test.csv')
test0=test0.set_index('Id')

train0_x=train0.iloc[:,:-1]
train0_y=train0.SalePrice
test0_x=test0.copy()

all_0=pd.concat([train0_x,test0_x])

#分定性和定量
qualitative_0=pd.DataFrame()
quantitative_0=pd.DataFrame()
for k in all_0.columns:
	dtype_k=all_0[k].dtype
	if dtype_k=='object':	#定性
		qualitative_0[k]=all_0[k]
	else:
		quantitative_0[k]=all_0[k]

qualitative_1 = qualitative_0.copy()
quantitative_1 = quantitative_0.copy()

#定性编码   
for k in qualitative_1.columns:  
	le=sk.preprocessing.LabelEncoder()
	qualitative_1[k][qualitative_1[k].notnull()]=le.fit_transform(qualitative_1[k][qualitative_1[k].notnull()])
#定性补缺       
imputer=Imputer(strategy='most_frequent')
qualitative_1 = pd.DataFrame(imputer.fit_transform(qualitative_1),index=qualitative_1.index)
#定性哑编码
encoder=OneHotEncoder(sparse=False,dtype=np.int)
qualitative_1 = pd.DataFrame(encoder.fit_transform(qualitative_1),index=qualitative_1.index)

#定量补缺
imputer=Imputer(strategy='mean')
quantitative_1 = pd.DataFrame(imputer.fit_transform(quantitative_1),index=quantitative_1.index)
#数据变换（升维）
pf=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
quantitative_1 = pd.DataFrame(pf.fit_transform(quantitative_1),index=quantitative_1.index)

#定性和定量合并
all_1=pd.concat([qualitative_1,quantitative_1],axis=1)
all_1.columns=np.arange(all_1.shape[1])

#分成train和test
train1_x=all_1.loc[train0_x.index,:]
test1_x=all_1.loc[test0_x.index,:]
train1_y=train0_y.copy()
#train1_y=train0_y.apply(float)/800000
#train1_y=train0_y.apply(float)
#判别分析（降维）
skb=SelectKBest(chi2,k=500)
all_2=skb.fit_transform(train1_x,train1_y)

########################
train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(all_2,train1_y,test_size=0.5)
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
            'max_depth':9
            }
num_rounds=3000
clf=xgb.train(params,dtrain,num_rounds,watchlist)
#self rate test
dtest_x_self=xgb.DMatrix(test_x)
test_y_pred=pd.Series(clf.predict(dtest_x_self),index=test_y_real.index)
#test_y_pred=test_y_pred*800000

print(math.sqrt(mean_squared_log_error(test_y_pred,test_y_real)))








#########################################
#########################################
#########################################
#########################################
#########################################



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
#not_0=train1_x.isnull().sum()[train1_x.isnull().sum()==0]
have_nan=train1_x.isnull().sum()[train1_x.isnull().sum()!=0]
test1_x[have_nan.index].fillna(0,inplace=True)
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
        
        #test1_x[i][test1_x[i].notnull()]=lbl.transform(test1_x[i][test1_x[i].notnull()])
        #test1_x[i] = test1_x[i].convert_objects(convert_numeric=True)

        #example=train1_x[i][train1_x[i].notnull()].values[0]
        #print(type(example))
'''       
sp=SelectPercentile()	#percentile=10
params=[1,2,5,10,15,20,25,30,35,40,50,60,70,80]
grid=GridSearchCV(sp,{'percentile':params})
train_x=grid.fit(train_x,train1_y).transform(train_x)
#test1_x=gus.transform(test1_x) 
'''


gus=GenericUnivariateSelect(param=20)
train_x=gus.fit_transform(train_x,train1_y) 
#test1_x=gus.transform(test1_x) 


'''
pca=PCA(n_components=30)
train_x=pca.fit_transform(train_x,train1_y) 
'''


########################

train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train_x,train1_y,test_size=0.5)

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
            'max_depth':9
            }
num_rounds=1000

#clf=xgb.train(params,dtrain,num_rounds,watchlist)
clf=xgb.train(params,dtrain,num_rounds,watchlist)


#self rate test
dtest_x_self=xgb.DMatrix(test_x)
test_y_pred=pd.Series(clf.predict(dtest_x_self),index=test_y_real.index)
#test_y_pred=test_y_pred*800000

print(math.sqrt(mean_squared_error(test_y_pred,test_y_real)))




'''
# real rate test
dtest1_x_self=xgb.DMatrix(test1_x)
test1_y_pred=pd.Series(clf.predict(dtest1_x_self),index=test1_x.index)*800000
result=pd.DataFrame(test1_y_pred,columns=['SalePrice'])
way_out='/media/m/文档/House Prices/data/result/'
result.to_csv(way_out+'result.csv')
'''
















