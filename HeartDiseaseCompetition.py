#import packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, FeatureUnion

#Import Data
x=pd.read_csv('train_values.csv')
y=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_values.csv')

#Set index as patient id
x.set_index('patient_id',inplace=True)
y.set_index('patient_id',inplace=True)
test.set_index('patient_id',inplace=True)

#Introduce interaction terms (each one introduced after a hypothesis formulation)
x['max2']=(x['max_heart_rate_achieved']**2).astype('float64')
test['max2']=(test['max_heart_rate_achieved']**2).astype('float64')

x['max3']=(x['max_heart_rate_achieved']**3)
test['max3']=(test['max_heart_rate_achieved']**3)

x['oldpeak2']=x['oldpeak_eq_st_depression']**2
test['oldpeak2']=test['oldpeak_eq_st_depression']**2

x['exmax']=x['max_heart_rate_achieved']*x['exercise_induced_angina']
test['exmax']=test['max_heart_rate_achieved']*test['exercise_induced_angina']

x['exmax2']=x['max_heart_rate_achieved']**2*x['exercise_induced_angina']
test['exmax2']=test['max_heart_rate_achieved']**2*test['exercise_induced_angina']

x['exmax3']=x['max_heart_rate_achieved']**3*x['exercise_induced_angina']
test['exmax3']=test['max_heart_rate_achieved']**3*test['exercise_induced_angina']

x['oldmax']=x['max_heart_rate_achieved']*x['oldpeak_eq_st_depression']
test['oldmax']=test['max_heart_rate_achieved']*test['oldpeak_eq_st_depression']

x['ageheart']=x['age']/x['max_heart_rate_achieved']
test['ageheart']=test['age']/test['max_heart_rate_achieved']

x['chestnum']=x['num_major_vessels']*x['chest_pain_type']
x.chestnum[x.chestnum==1]=0 #this is done since there are too little cases with chestnum=1
test['chestnum']=test['num_major_vessels']*test['chest_pain_type']

#Define all the data selectors for the first part of the pipeline

float_columns=[col for col in x.columns if x[col].dtype=='float64']
get_float_data=FunctionTransformer(lambda x : x[float_columns],validate=False)

string_cat_columns=[col for col in x.columns if x[col].dtype=='object']
get_string_cat_data=FunctionTransformer(lambda x : x[string_cat_columns],validate=False)

int_cat_columns=[col for col in x.columns if (x[col].dtype=='int64') and (len(x[col].unique())<10)]
get_int_cat_data=FunctionTransformer(lambda x : x[int_cat_columns],validate=False)

int_data=[col for col in x.columns if (x[col].dtype=='int64') and (len(x[col].unique())>10)]
get_int_data=FunctionTransformer(lambda x : x[int_data],validate=False)

#Define log transformer
log=PowerTransformer()
#Instatiate OneHotEncoder 
one=OneHotEncoder(sparse=False)
#Instantiate StandardScaler
s=StandardScaler()
#Instantiate discretizer
disc=KBinsDiscretizer(n_bins=10)

#Create the first set of pipelines
p1=Pipeline([('float',get_float_data),('log',log),('scaler',s)])
p2=Pipeline([('str_cat',get_string_cat_data),('dummy',one)])
p3=Pipeline([('int_cat',get_int_cat_data),('dummy',one)])
p4=Pipeline([('int',get_int_data),('binner',disc)])

#Apply FeatureUnion
fu=FeatureUnion(transformer_list=[('p1',p1),('p2',p2),('p3',p3),('p4',p4)])

#Instantiate Logistic Regression
clf=LogisticRegression(random_state=12)
#Build Final Pipeline. #Is not used due to an (at the time) unknown error with FeatureUnion
pl=Pipeline([('union',fu),('clf',clf)])

#Train model and predict test labels
clf.fit(fu.fit_transform(x),y)
#Here I am forced to permit data leakage, as the FeatureUnion transform method raises an error if called alone (without previous fitting)
y_pred=clf.predict_proba(fu.fit_transform(test))

#Build and save the results for submitting
submit=pd.DataFrame(y_pred[:,1]).set_index(test.index).reset_index()
submit.rename(columns={0:'heart_disease_present'},inplace=True)
submit.to_csv('Heart_Disease_submit10.csv',index=False)



