#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PowerTransformer
plt.style.use('seaborn')
from sklearn.pipeline import Pipeline, FeatureUnion
from SparseInteractions import SparseInteractions
import xgboost as xgb

#Import Data
x=pd.read_csv('train_values.csv')
y=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_values.csv')

#Set id to index
x.set_index('building_id',inplace=True)
y.set_index('building_id',inplace=True)
test.set_index('building_id',inplace=True)

def preprocesser(df):
    #Drop the columns that are considered not useful (a priori)
    x=df.copy()
    x.drop(x.columns[x.columns.str.startswith('has_secondary')],axis=1,inplace=True)

    x.drop('count_families',axis=1,inplace=True)

    x.drop('legal_ownership_status',axis=1,inplace=True)
    
    #Create new feature: Height-Area Ratio logarithm
    x['ha_ratio_log']=np.log(x.height_percentage/x.area_percentage)
    #Re-categorize some of the categorical data
    #count_floors
    x.count_floors_pre_eq[x.count_floors_pre_eq>4]=4
    #plan_configuration
    x.plan_configuration[x.plan_configuration.isin(['m','s','f'])]='msf'
    x.plan_configuration[np.logical_not(x.plan_configuration.isin(['msf','d','u','q']))]='other'
    #age
    labels=[0,1,2,3,4]
    x.age=pd.cut(x.age,bins=[-np.inf,10,20,30,50,np.inf],labels=labels).astype('int64')
    #geo_level_x_id
    labels=[0,1,2,3,4,5,6,7,8,9]
    x.geo_level_1_id=pd.cut(x.geo_level_1_id,bins=10,labels=labels).astype('int64')
    x.geo_level_2_id=pd.cut(x.geo_level_2_id,bins=10,labels=labels).astype('int64')
    x.geo_level_3_id=pd.cut(x.geo_level_3_id,bins=10,labels=labels).astype('int64')
    
    x.height_percentage=pd.cut(x.height_percentage,bins=10,labels=labels).astype('int64')
    x.area_percentage=pd.cut(x.height_percentage,bins=10,labels=labels).astype('int64')
    return x

#preprocess data
X=preprocesser(x)
Test=preprocesser(test)
#split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=42)

#Define all the data selectors for the first part of the pipeline

float_columns=[col for col in X.columns if X[col].dtype=='float64']
get_float_data=FunctionTransformer(lambda x : x[float_columns],validate=False)

string_cat_columns=[col for col in X.columns if (X[col].dtype=='object')]
get_string_cat_data=FunctionTransformer(lambda x : x[string_cat_columns],validate=False)

int_cat_columns=[col for col in X.columns if (X[col].dtype=='int64') and (len(X[col].unique())>2)]
get_int_cat_data=FunctionTransformer(lambda x : x[int_cat_columns],validate=False)

int_bin_columns=[col for col in X.columns if (X[col].dtype=='int64') and (len(X[col].unique())==2)]
get_int_bin_data=FunctionTransformer(lambda x : x[int_cat_columns],validate=False)

#Instatiate OneHotEncoder 
one=OneHotEncoder(sparse=True)
#Instantiate StandardScaler
s=StandardScaler()
#Instantiate  SparseInteractions
spar=SparseInteractions(degree=2) 

#Create the first set of pipelines
p1=Pipeline([('float',get_float_data),('scaler',s)])
p2=Pipeline([('str_cat',get_string_cat_data),('dummy',one)])
p3=Pipeline([('int_cat',get_int_cat_data),('dummy',one)])
p4=Pipeline([('int_bin',get_int_bin_data)])

#Apply FeatureUnion
fu=FeatureUnion(transformer_list=[('p1',p1),('p2',p2),('p3',p3),('p4',p4)])

#Instantiate the classifiers
clf=RandomForestClassifier(min_samples_leaf=5,n_estimators=1000,random_state=42,n_jobs=-1)
params = {
    "gamma": [0.7],
    "learning_rate": [0.5], # default 0.1 
    "max_depth": [6], # default 3
}
xgb_model=xgb.XGBClassifier(objective="multi:softmax", random_state=12,n_jobs=-1,min_samples_leaf=5,n_estimators=1000)
#grid_cv = GridSearchCV(xgb_model, param_grid = params, cv = 3,scoring='f1_micro')

#Build Final Pipeline
#pl=Pipeline([('union',fu),('clf',clf)])

#Train the model
clf.fit(fu.fit_transform(X),y)
#Make predictions
y_pred=clf.predict(fu.fit_transform(Test))

#print('The score is: '+str(f1_score(y_test,y_pred,average='micro')))


#Build and save the results for submitting
#y_pred=grid_cv.predict(fu.fit_transform(Test))
submit=pd.DataFrame(y_pred).set_index(Test.index).reset_index()
submit.rename(columns={0:'damage_grade'},inplace=True)
submit.to_csv('RichterSubmit5.csv',index=False)




