#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from IPython.display import Image
import pydotplus
#Import Data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

"""
#Basic Data exploration
########################################################################################################
df=pd.concat([train.drop(['casual','registered','count'],axis=1),test])
print(df.info())
print(df.describe())
   #missing values
print('Total number of missing values: '+str(train.isnull().sum().sum()+test.isnull().sum().sum()))
   #duplicated rows
print('Total number of duplicated values: '+str(train.duplicated().sum().sum()+test.duplicated().sum().sum()))
   #hist plot
df.drop(['datetime'],axis=1).hist()
plt.tight_layout()
plt.show()

#Time trends
train.set_index(pd.DatetimeIndex(train.datetime),inplace=True)
train.drop(['datetime'],axis=1,inplace=True)
    #Hour trend
fh, axes = plt.subplots(3,1)
sns.boxplot(data=train,y='count',x=train.index.hour,palette='Set1',ax=axes[0])
sns.boxplot(data=train,y='registered',x=train.index.hour,palette='Set1',ax=axes[1])
sns.boxplot(data=train,y='casual',x=train.index.hour,palette='Set1',ax=axes[2])
    #Day trend
fd, axes = plt.subplots(3,1)
sns.boxplot(data=train,y='count',x=train.index.weekday,palette='Set2',ax=axes[0])
sns.boxplot(data=train,y='registered',x=train.index.weekday,palette='Set2',ax=axes[1])
sns.boxplot(data=train,y='casual',x=train.index.weekday,palette='Set2',ax=axes[2])
    #Year trend
fy=plt.figure(4)
fy=sns.boxplot(data=train,y='count',x=train.index.year,palette='Set1')

#Weather trend (rain)
fw, axes = plt.subplots(3,1)
sns.boxplot(data=train,y='count',x=train.weather,palette='Set1',ax=axes[0])
sns.boxplot(data=train,y='registered',x=train.weather,palette='Set1',ax=axes[1])
sns.boxplot(data=train,y='casual',x=train.weather,palette='Set1',ax=axes[2])

#Continuous weather variables
corrtrain=train[['registered','casual','count','temp','atemp','humidity','windspeed']]
corr=corrtrain.corr()
#c=sns.heatmap(corr,cmap='coolwarm',linewidth=0.5) #heatmap
print(corr[['humidity','atemp','temp','windspeed']])
#windspeed is not correlated with the target variables and it will be not taken into account (only introduces noise)
"""

#Feature Engineering
##########################################################################

#We will treat casual and registered separetly
#Also we'll set the datetime stamp as the df index
traincasual=train.drop(['registered','count'],axis=1)
trainreg=train.drop(['casual','count'],axis=1)

traincasual.set_index(pd.DatetimeIndex(train.datetime),inplace=True)
traincasual.drop(['datetime'],axis=1,inplace=True)

trainreg.set_index(pd.DatetimeIndex(train.datetime),inplace=True)
trainreg.drop(['datetime'],axis=1,inplace=True)

test.set_index(pd.DatetimeIndex(test.datetime),inplace=True)

#Casual Feature Engineering

"""
    #Tree regressors binning
        #Hour Binning with decission tree
traincasual['Hour']=traincasual.index.hour
tree_casual_h=DecisionTreeRegressor(max_depth=3)
tree_casual_h.fit(traincasual.Hour.to_frame(),np.log(traincasual.casual+1))
            #extract hour thresholds from tree, get rid of negative values (leaves) and sort
tch_bins=np.sort(tree_casual_h.tree_.threshold[tree_casual_h.tree_.threshold>0])
            #append - and + infinity
bins=np.append(np.append(np.array([-np.inf]),tch_bins),np.array([np.inf]))
            #create the binned variable
labels=list(range(len(bins)-1))
        #Temperature binning  
tree_casual_t=DecisionTreeRegressor(max_depth=3)
tree_casual_t.fit(traincasual.atemp.to_frame(),np.log(traincasual.casual+1))
            #extract hour thresholds from tree, get rid of negative values (leaves) and sort
tct_bins=np.sort(tree_casual_t.tree_.threshold[tree_casual_t.tree_.threshold>0])
            #append - and + infinity
binst=np.append(np.append(np.array([-np.inf]),tct_bins),np.array([np.inf]))
            #bin names
labelst=list(range(len(binst)-1))
"""

def casual_prepro(traincasual):
    #Hour Binning
    traincasual['Hour']=traincasual.index.hour
    #traincasual['binHour']=pd.cut(traincasual.Hour,bins=bins,labels=labels)
    #Weekday and weekend addition
    traincasual['Weekday']=traincasual.index.weekday
    traincasual.loc[traincasual['Weekday']<5,'Weekend']=0
    traincasual.loc[traincasual['Weekday']>4,'Weekend']=1
    #Month variable
    traincasual['Month']=traincasual.index.month
    #Total period (two years) binning into 8 bins (quarterly) due to year tendency
    for j in range(traincasual.index.year.nunique()):
        y=traincasual.index.year==traincasual.index.year.unique()[j]
        q1=traincasual.index.month<4
        q2=(traincasual.index.month>=4) & (traincasual.index.month<7)
        q3=(traincasual.index.month>=7) & (traincasual.index.month<10)
        q4=(traincasual.index.month>=10)
        traincasual.loc[y & q1,'Quarter']=str(traincasual.index.year.unique()[j])+' 1'
        traincasual.loc[y & q2,'Quarter']=str(traincasual.index.year.unique()[j])+' 2'
        traincasual.loc[y & q3,'Quarter']=str(traincasual.index.year.unique()[j])+' 3'
        traincasual.loc[y & q4,'Quarter']=str(traincasual.index.year.unique()[j])+' 4'   
    #Temperature binning
    #traincasual['binAtemp']=pd.cut(traincasual.atemp,bins=len(labelst),labels=labelst)
    #Prepare data to create dummy variables for categorical variables.
    Xc=pd.concat([traincasual[['season','holiday','workingday','weather','Weekday','Weekend','Month','Quarter']].astype('category'),np.log(traincasual[['temp','atemp','humidity','Hour']]+1)],axis=1)
    Xcasual=pd.get_dummies(Xc,sparse=True)
    return Xcasual

Xtrain_casual=casual_prepro(traincasual)
Xtest_casual=casual_prepro(test)
ycasual=np.log(traincasual['casual']+1)

#Registered Feature Engineering

  #Tree regressor binning
"""
    #Hour Binning with decission tree
trainreg['Hour']=trainreg.index.hour
tree_reg_h=DecisionTreeRegressor(max_depth=3)
tree_reg_h.fit(trainreg.Hour.to_frame(),np.log(trainreg.registered+1))
        #extract hour thresholds from tree, get rid of negative values (leaves) and sort
trh_bins=np.sort(tree_reg_h.tree_.threshold[tree_reg_h.tree_.threshold>0])
        #append - and + infinity
binsr=np.append(np.append(np.array([-np.inf]),trh_bins),np.array([np.inf]))
        #create the binned variable
labelsr=list(range(len(binsr)-1))
    #Temperature binning  
tree_reg_t=DecisionTreeRegressor(max_depth=3)
tree_reg_t.fit(trainreg.atemp.to_frame(),np.log(trainreg.registered+1))
        #extract hour thresholds from tree, get rid of negative values (leaves) and sort
trt_bins=np.sort(tree_reg_t.tree_.threshold[tree_reg_t.tree_.threshold>0])
        #append - and + infinity
binsrt=np.append(np.append(np.array([-np.inf]),trt_bins),np.array([np.inf]))
        #bin names
labelsrt=list(range(len(binsrt)-1))
"""
def reg_prepro(trainreg):
    #Hour Binning with decission tree
    trainreg['Hour']=trainreg.index.hour
    #trainreg['binHour']=pd.cut(trainreg.Hour,bins=binsr,labels=labelsr)
    #Weekday and weekend addition
    trainreg['Weekday']=trainreg.index.weekday
    trainreg.loc[trainreg['Weekday']<5,'Weekend']=0
    trainreg.loc[trainreg['Weekday']>4,'Weekend']=1
    #Month variable
    trainreg['Month']=trainreg.index.month
    #Total period (two years) binning into 8 bins (quarterly) due to year tendency    
    for j in range(trainreg.index.year.nunique()):
        y=trainreg.index.year==trainreg.index.year.unique()[j]
        q1=trainreg.index.month<4
        q2=(trainreg.index.month>=4) & (trainreg.index.month<7)
        q3=(trainreg.index.month>=7) & (trainreg.index.month<10)
        q4=(trainreg.index.month>=10)
        trainreg.loc[y & q1,'Quarter']=str(trainreg.index.year.unique()[j])+' 1'
        trainreg.loc[y & q2,'Quarter']=str(trainreg.index.year.unique()[j])+' 2'
        trainreg.loc[y & q3,'Quarter']=str(trainreg.index.year.unique()[j])+' 3'
        trainreg.loc[y & q4,'Quarter']=str(trainreg.index.year.unique()[j])+' 4'
    #Temperature binning  
    #trainreg['binAtemp']=pd.cut(trainreg.atemp,bins=len(labelsrt),labels=labelsrt)
    #Dummy variables for categorical variables
    Xr=pd.concat([trainreg[['season','holiday','workingday','weather','Weekday','Weekend','Month','Quarter']].astype('category'),np.log(trainreg[['temp','atemp','humidity','Hour']]+1)],axis=1)
    Xreg=pd.get_dummies(Xr,sparse=True)
    return Xreg

Xtrain_reg=reg_prepro(trainreg)
Xtest_reg=reg_prepro(test)
yreg=np.log(trainreg['registered']+1)

#Model Building
##########################################################################################################################3

#Casual Model Building and prediction
modelcasual=RandomForestRegressor(n_estimators=120,random_state=123)
modelcasual.fit(Xtrain_casual,ycasual)
ycasual_pred=np.round(np.exp(modelcasual.predict(Xtest_casual)))-1

modelreg=RandomForestRegressor(n_estimators=120,random_state=123)
modelreg.fit(Xtrain_reg,yreg)
yreg_pred=np.round(np.exp(modelreg.predict(Xtest_reg)))-1

#Solution building and submission
ypred=pd.DataFrame(ycasual_pred+yreg_pred)
submit=ypred.set_index(test.index).reset_index()
submit.columns=['datetime','count']
submit.to_csv('BikeSharing1.csv',index=False)

'''
#For visualizing the tree
export_graphviz(tree_reg_h,out_file='trh.dot')
dot_data=export_graphviz(tree_reg_h,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png('trh.png')
plt.imshow(mpimg.imread('trh.png'))
'''
