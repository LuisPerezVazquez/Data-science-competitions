#import packages
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Import Data
########################################################################

#Building product names list
pn=['ahor','aval','cco','cder','cno','ctju','ctma','ctop','ctpp','deco','deme','dela','ecue','fond','hip','plan','pres','reca','tjcr','valo','viv']
prod_names=['ind_'+s+'_fin_ult1' for s in pn]+['ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
prod_fecha=prod_names+['fecha_dato','ncodpers']
 #RUN THIS BELOW
''' Only needs to be run once
def only_new_products(df,df_list,DF_list):
#Only select rows that have new products:
    #Change fecha_dato column to DateTime format
    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
    if len(df_list)>0:
        #Get all dataframes from the 'storing' list concatenated with df
        df_concat = pd.concat(df_list)
        del df_list[:] #delete df_list content to save memory
        df=pd.concat([df_concat,df])
    #keep rows with only one occurrence for next iteration
    df_list.append(df[df.ncodpers.map(df.ncodpers.value_counts()==1)])
    #select rows with more than 1 entry for clients
    df2=df[df.ncodpers.map(df.ncodpers.value_counts()>1)] 
    prev=df2.loc[:,prod_fecha]
    prev['fecha_dato']=prev['fecha_dato']+pd.DateOffset(months=1)  
    #SELF JOIN on MONTH=MONTH+1 and USER=USER (reset the index to keep it after the merge)
    self_join=df2.reset_index().merge(right=prev,suffixes=('','_prev'),how='left',left_on=['fecha_dato','ncodpers'],right_on=['fecha_dato','ncodpers']).set_index('index')
    #Create a subset of this DataFrame with the prev products
    prev_self=self_join[[p+'_prev' for p in prod_names]]
    prev_self.columns=self_join[prod_names].columns #setting the same column names so we can substract each dataframe directly
    #Create boolean array for wether or not ANY product was bought (difference with past month is 1)
    bool_prod=((self_join[prod_names].sub(prev_self))==1).any(axis=1)
    #bool_prod=((self_join[prod_names].sub(prev_self))==1).any(axis=1) | ((self_join[prod_names].sub(prev_self))==-1).any(axis=1)
    #Filter dataframe (filling bool index to match df with False )
    DF_list.append(df[bool_prod.reindex(df.index,fill_value=False)])
    #Keep newest entry of ncodpers for next iteration
       #df2.set_index(df2.fecha_dato,inplace=True)
    df_list.append(df2.reset_index().groupby('ncodpers').last().reset_index().set_index('index',drop=True)[df.columns]) #We can use .last() method because data is already ordered by date
    #df_list.append(df2.sort_values(by=['ncodpers','fecha_dato']).groupby('ncodpers').last().reset_index())
    #df_list.append(df[(~bool_prod).reindex(df.index,fill_value=False)].last('M').reset_index(drop=True))
    
    return df_list, DF_list

#To allow a faster data reading we load the files in chunks
chunks=pd.read_csv('train_ver2.csv',chunksize=1e6)
chunk_list=[]
store_list=[]
for chunk in chunks:
    store_list,chunk_list= only_new_products(chunk,store_list,chunk_list)
train=pd.concat(chunk_list)
#print(train.info()) we end up with a much smaller and useful dataframe
train.to_csv('train_new_prods.csv')
'''


#Read test file
test=pd.read_csv('test_ver2.csv')
'''Run Only Once
#Read 2016 May data to see which products each costumer already has (so we not recommend these to them again)
chunks_test=pd.read_csv('train_ver2.csv',chunksize=1e6)
chunk_tlist=[]
for chunk in chunks_test:
    chunk_tlist.append(chunk.loc[chunk.fecha_dato=='2016-05-28',prod_names+['ncodpers']])
    #I map every product entry 1 to 0 and 0 to 1, so I multiply this elementwise with the predicted probabilies matrix (see below in Model Building section), turning 0 probabilities of already owned products
may_prods=1-pd.concat(chunk_tlist) 
may_prods['ncodpers']=1-may_prods['ncodpers'] #get this back to norml
#print(set(test.ncodpers)-set(may_prods.ncodpers)) #there are apparently more entries in May (train) than in June (test), but luckily all June customers are in the May data
    #drop all May customers that are not in June test data 
extra_customers=[*(set(may_prods.ncodpers)-set(test.ncodpers)),]
may_prods_filtered = (may_prods.loc[~may_prods['ncodpers'].isin(extra_customers),:]).sort_values(by=['ncodpers'])
may_prods_filtered.to_csv('may_prods.csv')
'''
#Read simplified train file
train=pd.read_csv('train_new_prods.csv')
train.rename(columns={'Unnamed: 0':'index'},inplace=True)


#Data cleaning and exploring
########################################################################
  #We'll go column by column
#print(train.info())
#print(train.describe())
#print(train.isna().sum()) #to see the columns with missing values
#print([train[col].unique() for col in train.columns]) #see if there are some columns with weird values
#There are many columns with the same 364 missing values, we'll see if this data are the same rows
#if this were the case then this data we'll not be useful for the model and we'll drop it
#print(train.loc[train.ind_empleado.isna(),(train.columns[train.isna().sum()==364]).tolist()].isna().all()) #indeed they are the same
train=train[~train.ind_empleado.isna()]
  #Age
train.loc[(train.age==' NA')|(train.age.isna()),'age']=-1
train.age=train.age.astype(int)
#sns.distplot(train.age,axlabel='Age histogram')
  #Antiguedad 
#train.antiguedad.describe() #the minimum is -9999999 which doesn't make sense
train.loc[(train.antiguedad==-999999),'antiguedad']=0
train.antiguedad=train.antiguedad.astype(int)
  #Fecha_alta
#I'll redefine this feature so it makes more sense, calculating the difference of days between fecha_alta and the current date, as another type of antiguedad
train['fecha_alta']=(pd.to_datetime(train["fecha_dato"],format="%Y-%m-%d")-pd.to_datetime(train["fecha_alta"],format="%Y-%m-%d")).dt.days
  #Sexo
#print(train.groupby('sexo').count()) #fill the missing value with the most common genre
train.loc[train.sexo.isna(),'sexo']='V'
  #indrel_1mes
#print(train.groupby('indrel_1mes').count()) #fill the missing value with the most common 
train.loc[train.indrel_1mes.isna(),'indrel_1mes']=1
train.indrel_1mes=train.indrel_1mes.astype(int)
  #tiprel_1mes
#print(train.groupby('tiprel_1mes').count()) #fill the missing value with the most common 
train.loc[train.tiprel_1mes.isna(),'tiprel_1mes']='A'
  #tipodom only has 1 unique value, so useless
train.drop('tipodom',axis=1,inplace=True)
  #nomprov gives same information as cod_prov
train.drop('nomprov',axis=1,inplace=True)
  #codprov 
#print(train.cod_prov.unique())
#print(train.loc[~train.cod_prov.isna(),'cod_prov'].unique().max()) #asign the missing values to a new region: 53
train.loc[train.cod_prov.isna(),'cod_prov']=53 
train.cod_prov=train.cod_prov.astype(int)
  #renta
''' model training
#predict the missing values of renta based on age and cod_prov
bool_renta=(~(train.renta.isna()))&(train.age>17)&(train.age<101) #only use the data with ages between 18 and 100 for training, since the rest is quite noisy
train_renta=train.loc[bool_renta,['cod_prov','age']]
test_renta=train.loc[bool_renta,['renta']]
#treat cod_prov as numerical and not categorical to make the process faster (2 features against 54)
dict_sorted_renta=pd.Series(np.linspace(1,53,53,dtype=int),index=(train.groupby('cod_prov').median().renta).sort_values().index).to_dict()
train_renta.cod_prov=train_renta.cod_prov.map(dict_sorted_renta) #reassign cod_prov  to its rank according to renta (ordering the provincias by renta we'll make the learning easier)
renta_model=RandomForestRegressor(n_estimators=100)
renta_model.fit(train_renta,test_renta)
with open('renta_model.pkl', 'wb') as file:
    pickle.dump(renta_model, file)
'''
# Load from file
with open('renta_model.pkl', 'rb') as file:
    renta_model = pickle.load(file)
dict_sorted_renta=pd.Series(np.linspace(1,53,53,dtype=int),index=(train.groupby('cod_prov').median().renta).sort_values().index).to_dict()
renta_na=train.loc[train.renta.isna(),['cod_prov','age']]
renta_na['cod_prov_new']=renta_na.cod_prov.map(dict_sorted_renta)
renta_na['renta']=renta_model.predict(renta_na[['cod_prov_new','age']])

''' Sorry the plots are unpretty (but if you zoom you'll see that the predictions are pretty accurate) 
fa, axes=plt.subplots()
sns.factorplot(x='cod_prov',y='renta',data=train,ax=axes)
sns.factorplot(x='cod_prov',y='renta',data=renta_na,ax=axes)
fa, axes=plt.subplots()
sns.factorplot(x='cod_prov',y='renta',data=train,ax=axes)
sns.factorplot(x='cod_prov',y='renta',data=renta_na,ax=axes)
'''
train.loc[train.renta.isna(),['renta']]=renta_na.renta #assign predicted values to missing
  #canal_entrada
#asign the missing values to a new unknown channel: 'UKN'
train.loc[train.canal_entrada.isna(),'canal_entrada']='UKN'
  #conyuemp 
#This feature is 99.7% empty, perhaps due to a reading error, but as it does not seem to be so relevant, we'll just drop it for now
train.drop('conyuemp',axis=1,inplace=True)
  #ult_fec_cli_1t
'''
#This feature is also quite empty, but for different reasons: 
print(train.ult_fec_cli_1t.count()==(train.indrel==99).sum()) #we can see that ult_fec_cli_1t only records dates whenever a client stops having the primary client status (indrel=99)
#Therefore, the rest of data is empty just because the cient still has the primary status
#I'll redefine this feature so it makes more sense, calculating the difference in days with the current date (fecha_dato)
train['ult_fec_cli_1t']=(pd.to_datetime(train["fecha_dato"],format="%Y-%m-%d")-pd.to_datetime(train["ult_fec_cli_1t"],format="%Y-%m-%d")).dt.days
train.loc[train.ult_fec_cli_1t.isna(),'ult_fec_cli_1t']=0 #As discussed before if the value is missing is because the difference is 0
print(train.ult_fec_cli_1t.unique()) 
#There are a lot of negative values, which don't make any sense(the client stops being primary in the future!)
#This made me decide to drop the feature, since it is not well recorded (another option would be to set the negative values to 0)
'''
train.drop('ult_fec_cli_1t',axis=1,inplace=True)
  #segmento
#print(train.segmento.value_counts()) #There are to few na values as to create a category just for them
#sns.countplot(x='age',hue='segmento',data=train) #UNIVERSITARIOS (university students) are far more common in ages between 18 and 30, so I'll use this feature to input the missing values
train.loc[(train.segmento.isna())&(train.age>19)&(train.age<31),'segmento']='03 - UNIVERSITARIO'
train.loc[(train.segmento.isna())&~((train.age>19)&(train.age<31)),'segmento']='02 - PARTICULARES' #PARTICULARES is much more common than TOP, so I'll assign everything into the former. 
#Another option would be to look into 'renta' to separate them, but this difference is not that significant
  #ind_nomina_ult1 and ind_nom_pens_ult1         
train.loc[train.ind_nomina_ult1.isna(),'ind_nomina_ult1']=0
train.loc[train.ind_nom_pens_ult1.isna(),'ind_nom_pens_ult1']=0 #since they only have one missing value each we'll just input it as 0
#The cleaning in training data is done!
######################################
#Now let's see test data:
#print(test.isna().sum()) #to see the columns with missing values
  #Fecha_alta
#As with train data, redefine the variable as difference in days with current dates
test['fecha_alta']=(pd.to_datetime(test["fecha_dato"],format="%Y-%m-%d")-pd.to_datetime(test["fecha_alta"],format="%Y-%m-%d")).dt.days
  #Antiguedad
test.antiguedad=test.antiguedad.astype(int)
#test.antiguedad.describe() #the minimum is -9999999 which doesn't make sense
test.loc[(test.antiguedad==-999999),'antiguedad']=0
  #Drop fecha_dato (only one unique value)
test.drop('fecha_dato',axis=1,inplace=True)
  #Drop ult_fec_cli_1t and conyuemp (almost all missing values)
test.drop('conyuemp',axis=1,inplace=True)
test.drop('ult_fec_cli_1t',axis=1,inplace=True)
  #Sexo
test.loc[test.sexo.isna(),'sexo']='V'
  #tipodom
test.drop('tipodom',axis=1,inplace=True) #only one unique value, so useless
  #indrel_1mes
#print(test.groupby('indrel_1mes').count()) #fill the missing value with the most common 
test.loc[test.indrel_1mes.isna(),'indrel_1mes']=1
test.indrel_1mes=test.indrel_1mes.astype(int)
  #tiprel_1mes
#print(test.groupby('tiprel_1mes').count()) #fill the missing value with the most common 
test.loc[test.tiprel_1mes.isna(),'tiprel_1mes']='I'
  #nomprov gives same information as cod_prov
test.drop('nomprov',axis=1,inplace=True)
  #codprov 
#print(test.cod_prov.unique())
#print(test.loc[~test.cod_prov.isna(),'cod_prov'].unique().max()) #asign the missing values to a new region: 53
test.loc[test.cod_prov.isna(),'cod_prov']=53 
test.cod_prov=test.cod_prov.astype(int)
  #canal_entrada
#asign the missing values to a new unknown channel: 'UKN'
test.loc[test.canal_entrada.isna(),'canal_entrada']='UKN'
  #segmento
#sns.countplot(x='age',hue='segmento',data=test) same as with train data
test.loc[(test.segmento.isna())&(test.age>19)&(test.age<32),'segmento']='03 - UNIVERSITARIO'
test.loc[(test.segmento.isna())&~((test.age>19)&(test.age<32)),'segmento']='02 - PARTICULARES'
  #renta
test.loc[test.renta=='         NA','renta']=np.nan #there are 227965 missing values for renta
test.renta=test.renta.astype(float)
#Use the model trained before with the train data to fill the missing values
renta_na_test=test.loc[test.renta.isna(),['cod_prov','age']]
renta_na_test['cod_prov_new']=renta_na_test.cod_prov.map(dict_sorted_renta)
renta_na_test['renta']=renta_model.predict(renta_na_test[['cod_prov_new','age']])
test.loc[test.renta.isna(),['renta']]=renta_na_test.renta #assign predicted values to missing
###################################
#Target variables
pn=['ahor','aval','cco','cder','cno','ctju','ctma','ctop','ctpp','deco','deme','dela','ecue','fond','hip','plan','pres','reca','tjcr','valo','viv']
prod_names=['ind_'+s+'_fin_ult1' for s in pn]+['ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
prod_fecha=prod_names+['fecha_dato']
  #Plot products added by month (normalized)
#prods_month=train[prod_fecha].groupby(by='fecha_dato').sum()  #total amount of each product purchased each month
#tot_prods_month=train[prod_fecha].groupby(by='fecha_dato').sum().sum(axis=1) #total products purchased by month
#((prods_month).divide(tot_prods_month,axis='index')).plot(kind='bar',stacked=True)
  #Plot total products purchased
#train.loc[train.fecha_dato=='2015-06-28',prod_names].sum().sort_values().plot(kind='bar')
#print(train[prod_names].sum()) ind_ahor_fin_ult1 and ind_aval_fin_ult1 are extremly rare and will be therefore dropped (cder,ctju,pres,deme <1000)
train.drop('ind_ahor_fin_ult1',axis=1,inplace=True)
train.drop('ind_aval_fin_ult1',axis=1,inplace=True)

#Feature engineering and Model Building
########################################################################
#First model: Seasonality and some trend
#Since my computer can't train the model on all the data, 
#I'll train the model only on the data we have of the same month as the test data: June,
#And two other months to capture the trend, December 2015 and May 2016
#The model will consist of individual xgboost binary classifiers (0 or 1) for each product
   #Define target and training features
pn=['cco','cder','cno','ctju','ctma','ctop','ctpp','deco','deme','dela','ecue','fond','hip','plan','pres','reca','tjcr','valo','viv']
prod_names=['ind_'+s+'_fin_ult1' for s in pn]+['ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
feature_names=train.columns[['_ult1' not in col for col in train.columns]].to_list()
   #Select only june data
train_june=train.loc[(train.fecha_dato=='2015-06-28')|(train.fecha_dato=='2015-12-28')|(train.fecha_dato=='2016-05-28'),feature_names]
   #Prepare data for training model (test and train data)
test_sorted=test.sort_values(by=['ncodpers'])
Xtest=pd.concat([test_sorted[['ind_empleado','pais_residencia','sexo','ind_nuevo','indrel','indrel_1mes','tiprel_1mes','indresi','indext','canal_entrada','indfall','cod_prov','ind_actividad_cliente','segmento']].astype('category'),test_sorted[['age','fecha_alta','antiguedad','renta']]],axis=1)
Xtrain=pd.concat([train_june[['ind_empleado','pais_residencia','sexo','ind_nuevo','indrel','indrel_1mes','tiprel_1mes','indresi','indext','canal_entrada','indfall','cod_prov','ind_actividad_cliente','segmento']].astype('category'),train_june[['age','fecha_alta','antiguedad','renta']]],axis=1)
#assign a random float number to separate after the 'dummyfication'
Xtrain['type']=0.1
Xtest['type']=1.1 
#get dummies on the concat of both data sets to have same number of categories
X_dummies=pd.get_dummies(pd.concat([Xtest,Xtrain]).drop(['canal_entrada','pais_residencia'],axis=1))
Xtrain_dummies=X_dummies.loc[X_dummies.type==0.1,:]
Xtest_dummies=X_dummies.loc[X_dummies.type==1.1,:]
Xtrain_dummies.drop('type',axis=1,inplace=True)
Xtest_dummies.drop('type',axis=1,inplace=True)
ytrain=[train.loc[(train.fecha_dato=='2015-06-28')|(train.fecha_dato=='2015-12-28')|(train.fecha_dato=='2016-05-28'),col] for col in prod_names] #create a target variable for each product
xgb_models=[xgb.XGBClassifier(objective='binary:logistic').fit(Xtrain_dummies,y) for y in ytrain] #train a model for each product
#xgb_models=[RandomForestClassifier(n_estimators=120,random_state=123).fit(Xtrain_dummies,y) for y in ytrain] #train a model for each product

#Xtest is too big to predict probabilities all at once, so we'll process it in chunks
predict_list=[]
for chunk in np.array_split(Xtest_dummies,100):
    xgb_predicted_proba=[model.predict_proba(chunk)[:,1] for model in xgb_models]
    predict_list.append(pd.DataFrame(xgb_predicted_proba).T)   
prods_proba=pd.concat(predict_list) #concat obtained list
prods_proba.columns=prod_names #rename columns as the product names
#Map probabilities of already owned products to 0 (I'll use the previously calculated matrix may_prods_filtered, see above in Import Data section)
mult_may=pd.read_csv('may_prods.csv')
no_owned_prods_proba=mult_may[prod_names].reset_index(drop=True).mul(prods_proba.reset_index(drop=True))

#Now we have to extract the top 7 most probable products from each row, we'll define a function for this
def find_top_n(df,n):
    top_n=df.apply(lambda row, n: pd.Series(row.nlargest(n).index), axis=1, n=n) #on each row, gets the index (column name) of the 7 top probabilities
    return top_n
recommendations=find_top_n(no_owned_prods_proba,7) #As expected takes a while
#Add cuestomer id and rearrange according to test data original order
sub=pd.concat([test_sorted.reset_index(drop=True).ncodpers,recommendations],axis=1).set_index('ncodpers').reindex(test.set_index('ncodpers').index)
submission_june=pd.DataFrame()
submission_june['added_products']=sub.iloc[:,0]+' '+sub.iloc[:,1]+' '+sub.iloc[:,2]+' '+sub.iloc[:,3]+' '+sub.iloc[:,4]+' '+sub.iloc[:,5]+' '+sub.iloc[:,6]
submission_june.reset_index().to_csv('june_recom2.csv',index=False)
  
