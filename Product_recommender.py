
# coding: utf-8

# # Santander Product Recommendation
# 
# <img src="https://www.santander.co.uk/themes/custom/santander_web18/logo.svg" alt="Alt text that describes the graphic"/>
# 
# 
# #### By
# ## Mohit Joshi 
# ### Manager, Data Science and Engineering
# 

# Under their current system, a small number of Santander’s customers receive many recommendations while many others rarely see any resulting in an uneven customer experience. In their second competition, Santander is challenging Kagglers to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.
# 
# With a more effective recommendation system in place, Santander can better meet the individual needs of all customers and ensure their satisfaction no matter where they are in life.

# <b>MODELLING STRATEGY</b>![image.png](attachment:image.png)

# ## Load the Training Data

# In[1]:


import numpy as np
import pandas as pd
import plotly
#import plotly.plotly as py
from plotly.offline import plot, iplot, init_notebook_mode
from plotly import graph_objs as go
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import accuracy_score
import xgboost as xgb
init_notebook_mode(connected=True)


# In[128]:


#Set the enviornment to make the visual better

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', -1)

# read_init_df = pd.read_csv('santander-product-recommendation/train_ver2.csv')
redd_init_df = pd.read_csv('santander-product-recommendation/train_ver2.csv')


# In[3]:


# I have checked the dataset and we have following products for which we need to do the prediction

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']


# In[4]:


#Lets do a dive into dataset 

redd_init_df.head(10)

# We have a lot of columns here which can help us in creating a better model


# ## Load the column description
# 
# I will be using this table to take right decision with respect to columns

# In[5]:


# I have created a file for my own convenience to check the definition of the column
# Lets import the file 
column_desc = pd.read_excel('santander-product-recommendation/col_desc.xlsx')


# In[6]:


# this view will help me throughout the exercise, to make the decision in 'adding' or 'dropping' a dataset from the table
column_desc.head(50)


# #### Let's change the date format to start with Exploratory data analysis

# In[129]:


# We can see from the definition that we have three different date type column, will convert then into more readable
# and executionable format 

redd_init_df.fecha_dato = pd.to_datetime(redd_init_df.fecha_dato, format="%Y-%m-%d")
redd_init_df.fecha_alta = pd.to_datetime(redd_init_df.fecha_alta, format="%Y-%m-%d")
redd_init_df.ult_fec_cli_1t = pd.to_datetime(redd_init_df.ult_fec_cli_1t, format="%Y-%m-%d")


# ### Data Exploration

# In[149]:


# checking the distrubution of the data
# As we can see the data volume is huge and it will take a lot of time to do every task

redd_init_df.describe()


# In[9]:


# Here i am defining a function to check the null percenatge value in the dataframe loaded

def check_null_per(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    return missing_value_df


# In[10]:


#Lets check the initial null values across the columns

check_null_per(redd_init_df)


# In[11]:


# Dropping the columns with more than 90% missing data i.e 'ult_fec_cli_1t' and 'conyuemp'

Drop_col = ['ult_fec_cli_1t','conyuemp']
try:
    redd_init_df.drop(Drop_col, axis=1, inplace=True)
except Exception as e:
    print (e)

#check_null_per(redd_init_df)


# In[150]:


age_count = redd_init_df.groupby('age')['ncodpers'].nunique().reset_index()
age_count.head(10)
#def get_salary_average(Age):


# ### Let's check the Age vs Customers distribution

# In[151]:


data = []


    
data = [
go.Bar(
    x=age_count['age'], # assign x as the dataframe column 'x'
    y=age_count['ncodpers'])
]

layout = go.Layout(
    title='No of customers from different age groups'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# ### Let's check the Monthly sold product for the entire dataset
# 
# This will help us identifying the behaviour of each product better

# In[145]:


redd_init_prod = redd_init_df.groupby('fecha_dato')[product_col].sum().reset_index()

data = []

for product in product_col:
    temp = go.Scatter(
            x=redd_init_prod['fecha_dato'], # assign x as the dataframe column 'x'
            y=redd_init_prod[product], 
            mode = 'lines', name = product
            )        
    data.append(temp)

layout = go.Layout(
    title='Product Sold over several months'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# ### Product sold vs Age
# 
# Lets' see which group of age is buying the most of the product

# In[148]:


redd_init_age_prod = redd_init_df.groupby('age')[product_col].sum().reset_index()

data = []

for product in product_col:
    temp = go.Scatter(
            x=redd_init_age_prod['age'], # assign x as the dataframe column 'x'
            y=redd_init_age_prod[product], 
            mode = 'lines', name = product
            )        
    data.append(temp)

layout = go.Layout(
    title='Product Sold for different age groups over several months'
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


# ## DATA CLEANING

# In[14]:


# DATA CLEANING

# We will be deleting the Rows where the sex of the person is not present, also for 'ind_nomina_ult1' and 'ind_nom_pens_ult1'

redd_init_df = redd_init_df[pd.notnull(redd_init_df['sexo'])]

redd_init_df = redd_init_df[pd.notnull(redd_init_df['ind_nomina_ult1'])]
redd_init_df = redd_init_df[pd.notnull(redd_init_df['ind_nom_pens_ult1'])]

#Get the dataset for which age is less than or equal to 100

redd_init_df = redd_init_df[redd_init_df.age <= 100]

# We will be replacing null values in 'indrel_1mes' 'tiprel_1mes' columsn with 'Others' to give a new type when 
# there is no string

redd_init_df.indrel_1mes.fillna("Others", inplace = True)
redd_init_df.tiprel_1mes.fillna("Others",inplace = True)

# We will replace column 'canal_entrada' with mode of the the column, since it is channel used by the customer so makes
# more sense to use the channel which has highest frequency of usage

redd_init_df.canal_entrada.fillna(redd_init_df['canal_entrada'].mode()[0], inplace = True)



# In[15]:


# As we can see the column cod_Prov and nom_Prov has some null values still, we will replace it with NA, as the this information
# cannot be imputed

redd_init_df.cod_prov.fillna("NA", inplace = True)
redd_init_df.nomprov.fillna("NA",inplace = True)

#Replacing null values in Segmaneto column with the mode

redd_init_df.segmento.fillna(redd_init_df.segmento.mode()[0],inplace = True)

redd_init_df.renta.fillna("NA",inplace = True)

# redd_init_df.tipodom.value_counts()
redd_init_df.tipodom.fillna('1.0',inplace = True)


# In[16]:


check_null_per(redd_init_df)


# In[17]:


redd_init_df.indrel_1mes.value_counts()


# In[18]:


# As you can see there are multiple ways in which the same value has been entered
# We will enter the string value which has been given and later use One-hot encoding

redd_init_df.loc[redd_init_df.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
redd_init_df.loc[redd_init_df.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
redd_init_df.loc[redd_init_df.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
redd_init_df.loc[redd_init_df.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'

redd_init_df.loc[redd_init_df.indrel_1mes == '2', 'indrel_1mes'] = 'CoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == '2.0', 'indrel_1mes'] = 'CoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == 2, 'indrel_1mes'] = 'CoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == 2.0, 'indrel_1mes'] = 'CoOwner'

redd_init_df.loc[redd_init_df.indrel_1mes == '3', 'indrel_1mes'] = 'FormerPrimary'
redd_init_df.loc[redd_init_df.indrel_1mes == '3.0', 'indrel_1mes'] = 'FormerPrimary'
redd_init_df.loc[redd_init_df.indrel_1mes == 3, 'indrel_1mes'] = 'FormerPrimary'
redd_init_df.loc[redd_init_df.indrel_1mes == 3.0, 'indrel_1mes'] = 'FormerPrimary'

redd_init_df.loc[redd_init_df.indrel_1mes == '4', 'indrel_1mes'] = 'FormerCoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == '4.0', 'indrel_1mes'] = 'FormerCoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == 4, 'indrel_1mes'] = 'FormerCoOwner'
redd_init_df.loc[redd_init_df.indrel_1mes == 4.0, 'indrel_1mes'] = 'FormerCoOwner'

redd_init_df.loc[redd_init_df.indrel_1mes == "P", 'indrel_1mes'] = 'Others'


# In[19]:


redd_init_df.indrel_1mes.value_counts()


# In[20]:


#We will do some typecasting now to make out data more accurate, redable and able to predict better 

#redd_init_df.segmento.value_counts()

redd_init_df['ind_nuevo'] = redd_init_df['ind_nuevo'].astype(np.uint8)
redd_init_df['ind_actividad_cliente'] = redd_init_df['ind_actividad_cliente'].astype(np.uint8)
redd_init_df['antiguedad'] = redd_init_df['antiguedad'].astype(np.int32)    
redd_init_df['renta'] = redd_init_df['renta'].replace('NA',0).astype(np.float64)
redd_init_df['age'] = redd_init_df['age'].astype(np.float64)


# In[21]:


# Defining the datatype for all the products, we will be using these products values to calculate extra features 
# in future

for col in product_col:
            redd_init_df[col] = redd_init_df[col].astype(np.uint8)


# In[24]:


# redd_init_df.head(10)

redd_init_df[redd_init_df['fecha_dato'] == '2015-07-28']

# redd_init_df.fecha_dato.value_counts()


# In[25]:


## We will be defining a list of dates which will be used for training and testing purpose here

train_dates = ['2015-07-28','2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28',                '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

# train_dates = ['2016-03-28', '2016-04-28', '2016-05-28']

test_dates = ['2016-06-28']


# In[31]:


redd_init_df.shape


# ## FEATURE ENGINEERING

# In[26]:


# Feature Engineering


# Create a different dataframe for all the dates

dates = train_dates + test_dates
all_dataframe = {}

for dt in dates:
    if dt in train_dates:
        all_dataframe['redd_init_'+dt] = redd_init_df[redd_init_df['fecha_dato'] == dt]
    else:
        all_dataframe['redd_init_'+dt] = redd_init_df[redd_init_df['fecha_dato'] == train_dates[0]]


# In[27]:


# Create Feature for every separate dataframe to get the details of how many products have they bought in previous
# months

# df = all_dataframe['redd_init_2015-07-28']

def get_product_sum(df, lag):        
    z = [ row.ind_ahor_fin_ult1 + row.ind_aval_fin_ult1+row.ind_cco_fin_ult1+row.ind_cder_fin_ult1+         row.ind_cno_fin_ult1 + row.ind_ctju_fin_ult1+row.ind_ctma_fin_ult1+ row.ind_ctop_fin_ult1+         row.ind_ctpp_fin_ult1+row.ind_deco_fin_ult1+row.ind_deme_fin_ult1+row.ind_dela_fin_ult1+row.ind_ecue_fin_ult1+         row.ind_fond_fin_ult1+row.ind_hip_fin_ult1+row.ind_plan_fin_ult1+row.ind_pres_fin_ult1+row.ind_reca_fin_ult1+         row.ind_tjcr_fin_ult1+row.ind_valo_fin_ult1+row.ind_viv_fin_ult1+row.ind_nomina_ult1+row.ind_nom_pens_ult1+         row.ind_recibo_ult1 for index, row in df.iterrows()]
    df[lag] = z
    return df
    
for item, value in all_dataframe.items():
    df = all_dataframe[item]
    temp_df = get_product_sum(df, 'Product_sum')
    all_dataframe[item] = temp_df


# In[87]:


## Add lag columns in all the dataframes separated by dates

for dt in dates:
    df = all_dataframe['redd_init_'+str(dt)]
    df['Product_sum_lag1'] = ''
    df['Product_sum_lag2'] = ''
    df['Product_sum_lag3'] = ''
    all_dataframe['redd_init_'+str(dt)] = df
    


# In[144]:



# Create a column in all the dataframes separated by dates to calculate the number of products bought in last 1 months

for i in range(len(dates)):
    print i
    try:
        df = all_dataframe['redd_init_'+str(dates[i+1])]
        df_previous = all_dataframe['redd_init_'+str(dates[i])]
        df_join = df.merge(df_previous[['ncodpers','Product_sum']],on='ncodpers',how='left',suffixes=("","_lag"))
        df = df.copy()
        df = df.set_index(df_join.index)
        df['Product_sum_lag1'] = df_join['Product_sum_lag']
        df = df[pd.notnull(df['fecha_dato'])]
        all_dataframe['redd_init_'+str(dates[i+1])] = df
    except Exception as e:
        print e


# In[150]:


# Create a column in all the dataframes separated by dates to calculate the number of products bought in last 2 months
# This is a lag feature in order to use it for future prediction 

for i in range(len(dates)):
    print i
    try:
        df = all_dataframe['redd_init_'+str(dates[i+2])]
        df_previous = all_dataframe['redd_init_'+str(dates[i])]
        df_join = df.merge(df_previous[['ncodpers','Product_sum']],on='ncodpers',how='left',suffixes=("","_lag"))
        df = df.copy()
        df_join = df_join.set_index(df.index)
        df['Product_sum_lag2'] = df_join['Product_sum_lag']
        all_dataframe['redd_init_'+str(dates[i+2])] = df
    except Exception as e:
        print e


# In[156]:


backup = all_dataframe


# In[151]:


# Create a column in all the dataframes separated by dates to calculate the number of products bought in last 3 months
for i in range(len(dates)):
    print i
    try:
        df = all_dataframe['redd_init_'+str(dates[i+3])]
        df_previous = all_dataframe['redd_init_'+str(dates[i])]
        df_join = df.merge(df_previous[['ncodpers','Product_sum']],on='ncodpers',how='left',suffixes=("","_lag"))
        df = df.copy()
        df_join = df_join.set_index(df.index)
        df['Product_sum_lag3'] = df_join['Product_sum_lag']
        all_dataframe['redd_init_'+str(dates[i+3])] = df
    except Exception as e:
        print e


# In[198]:


# Product count in lag for all the customers
# all_dataframe = backup

for a in dates:
    print a
    df = all_dataframe['redd_init_'+str(a)]
    temp_df = redd_init_df[redd_init_df.fecha_dato < a].groupby('ncodpers',as_index=False)[product_col].sum()
    temp_df.ncodpers = temp_df.ncodpers.astype(int)
    df = df.merge(temp_df,on='ncodpers',how='left',suffixes=("","_lagproductcount"))
#     print df.head()
# #     print df
    all_dataframe['redd_init_'+str(a)] = df


# ### CHECKING AND TESTING THE DATA

# In[201]:


# all_dataframe = backup
# all_dataframe['redd_init_2016-06-28']

# df[df.fecha_dato < a].groupby('ncodpers',as_index=False)[product_col].sum()

# dropc = [u'ind_ahor_fin_ult1_lagproductcount',
#        u'ind_aval_fin_ult1_lagproductcount',
#        u'ind_cco_fin_ult1_lagproductcount',
#        u'ind_cder_fin_ult1_lagproductcount',
#        u'ind_cno_fin_ult1_lagproductcount',
#        u'ind_ctju_fin_ult1_lagproductcount',
#        u'ind_ctma_fin_ult1_lagproductcount',
#        u'ind_ctop_fin_ult1_lagproductcount',
#        u'ind_ctpp_fin_ult1_lagproductcount',
#        u'ind_deco_fin_ult1_lagproductcount',
#        u'ind_deme_fin_ult1_lagproductcount',
#        u'ind_dela_fin_ult1_lagproductcount',
#        u'ind_ecue_fin_ult1_lagproductcount',
#        u'ind_fond_fin_ult1_lagproductcount',
#        u'ind_hip_fin_ult1_lagproductcount',
#        u'ind_plan_fin_ult1_lagproductcount',
#        u'ind_pres_fin_ult1_lagproductcount',
#        u'ind_reca_fin_ult1_lagproductcount',
#        u'ind_tjcr_fin_ult1_lagproductcount',
#        u'ind_valo_fin_ult1_lagproductcount',
#        u'ind_viv_fin_ult1_lagproductcount', u'ind_nomina_ult1_lagproductcount',
#        u'ind_nom_pens_ult1_lagproductcount',
#        u'ind_recibo_ult1_lagproductcount',
#        u'ind_ahor_fin_ult1_lagproductcount',
#        u'ind_aval_fin_ult1_lagproductcount',
#        u'ind_cco_fin_ult1_lagproductcount',
#        u'ind_cder_fin_ult1_lagproductcount',
#        u'ind_cno_fin_ult1_lagproductcount',
#        u'ind_ctju_fin_ult1_lagproductcount',
#        u'ind_ctma_fin_ult1_lagproductcount',
#        u'ind_ctop_fin_ult1_lagproductcount',
#        u'ind_ctpp_fin_ult1_lagproductcount',
#        u'ind_deco_fin_ult1_lagproductcount',
#        u'ind_deme_fin_ult1_lagproductcount',
#        u'ind_dela_fin_ult1_lagproductcount',
#        u'ind_ecue_fin_ult1_lagproductcount',
#        u'ind_fond_fin_ult1_lagproductcount',
#        u'ind_hip_fin_ult1_lagproductcount',
#        u'ind_plan_fin_ult1_lagproductcount',
#        u'ind_pres_fin_ult1_lagproductcount',
#        u'ind_reca_fin_ult1_lagproductcount',
#        u'ind_tjcr_fin_ult1_lagproductcount',
#        u'ind_valo_fin_ult1_lagproductcount',
#        u'ind_viv_fin_ult1_lagproductcount', u'ind_nomina_ult1_lagproductcount',
#        u'ind_nom_pens_ult1_lagproductcount',
#        u'ind_recibo_ult1_lagproductcount']

# for a in dates:
#     df = all_dataframe['redd_init_'+str(a)]
#     df = df.drop(dropc,axis = 1)
#     all_dataframe['redd_init_'+str(a)] = df


# ### Create a checkpoint here to save the data frame after feature engineering

# In[204]:


final_df = pd.DataFrame()
for a in dates:
    df = all_dataframe['redd_init_'+str(a)]
    final_df = final_df.append(df,ignore_index=True)


# In[212]:


# final_df

final_df.to_csv(r'~/final_df.csv')


# ### Start from where you left it off

# In[28]:


final_df = pd.read_csv('~/final_df.csv')


# #### More celaning

# In[29]:


final_df.fillna(0, inplace = True)
final_df['Product_sum_lag1'] = final_df['Product_sum_lag1'].replace('',0).astype(int)
final_df['Product_sum_lag2'] = final_df['Product_sum_lag2'].replace('',0).astype(int)
final_df['Product_sum_lag3'] = final_df['Product_sum_lag3'].replace('',0).astype(int)

final_df.fecha_dato = pd.to_datetime(final_df.fecha_dato, format="%Y-%m-%d")

final_df['fyear'] = final_df['fecha_dato'].dt.year
final_df['fmonth'] = final_df['fecha_dato'].dt.month
final_df['fday'] = final_df['fecha_dato'].dt.day


# In[30]:


final_df.shape


# In[35]:


target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']


# ### Feature Engineering for adding a Label to the dataset
# 
# Although we have all the product columns which can be used for modelling exercise, we will be creating a new feature called product whhich will tell for a customer which product he/ she bought and whether it is newly added since last month
# 
# Basically, the modelling can be done two way here:
# 
# > One, where we can create multiple models by treating every product as target variable and aggregating it at the end
# 
# 
# > Two, The way we are doing it in this notebook

# In[36]:


#     #Select only the target columns.

train_dates = ['2015-07-28','2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28',                '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']

# train_dates = ['2016-03-28', '2016-04-28', '2016-05-28']

test_dates = ['2016-06-28']

final_label_df = pd.DataFrame()

dates = train_dates + test_dates

for i in range(len(dates)):
    try:
        print i
        train_new_month = final_df[final_df['fecha_dato']== dates[i+1]]
        print train_new_month.shape
        train_previous_month = final_df[final_df['fecha_dato']== dates[i]]
        train_new_month_targets = train_new_month[target_cols]
        #Add ncodpers to the dataframe.
        train_new_month_targets['ncodpers'] = train_new_month.ncodpers#Remove the index.
        train_new_month_targets.reset_index(drop = True, inplace = True)
        #Select only the target columns.
        train_previous_month_targets = train_previous_month[target_cols]
        #Add ncodpers to the dataframe.
        train_previous_month_targets['ncodpers'] = train_previous_month.ncodpers
        #Set ncodpers' values to 0, so that there is no effect to this feature when this dataframe is 
        #subtracted from train_new_month_targets.
        train_previous_month_targets.ncodpers = 0
        #Remove the index.
        train_previous_month_targets.reset_index(drop = True, inplace = True)
        #Subtract the previous month from the current to find which new products the customers have.
        train_new_products = train_new_month_targets.subtract(train_previous_month_targets)
        #Values will be negative if the customer no longer has a product that they once did. 
        #Set these negative values to 0.
        train_new_products[train_new_products < 0] = 0
        train_new_products = train_new_products.fillna(0)
        #Merge the target features with the data we will use to train the model.
        train_new_products = train_new_products.merge(train_new_month.ix[:,0:23], on = 'ncodpers')
        train_final = train_new_products
        labels = train_final[target_cols]
        labels['ncodpers'] = train_final.ncodpers
        labels = labels.set_index("ncodpers")
        stacked_labels = labels.stack()
        filtered_labels = stacked_labels.reset_index()
        filtered_labels.columns = ["ncodpers", "product", "newly_added"]
        filtered_labels = filtered_labels[filtered_labels["newly_added"] == 1]
        filtered_labels = filtered_labels.drop_duplicates(subset=['ncodpers'], keep='last')
        train_new_month = train_new_month.merge(filtered_labels, on="ncodpers", how="left" )
        final_label_df = final_label_df.append(train_new_month,ignore_index=True)
        print final_label_df.shape
    except Exception as e:
        print e



# In[37]:


final_df1 = final_label_df[final_label_df['newly_added'] == 1] 


# In[38]:


final_df1['product'] = final_df1['product'].astype('category').cat.codes


# In[49]:


final_df1 = final_df1.drop(['Unnamed: 0'], axis = 1)
final_df1.shape


# ### Get the different types of columns required for modelling 
# 
# Here we will have 
# 
#     Training columns
#     Label column
#     date column
#     Categorical columns
#     Non-categorical columns

# In[50]:


date_cols = ['fecha_dato']

label_cols = ['product']

trainingcols = list(set(final_df1.columns)-set(target_cols)- set(date_cols)- set(label_cols))


# In[51]:


cat_cols = ['indfall', 'indresi', 'pais_residencia', 'indext', 'segmento', 'canal_entrada', 'tiprel_1mes', 'fecha_alta',           'indrel_1mes', 'sexo', 'nomprov', 'cod_prov', 'ind_empleado']

non_cat_cols = list(set(trainingcols)-set(cat_cols))

tmp = []


# In[52]:


df = pd.DataFrame({col: final_df1[col].astype('category').cat.codes for col in cat_cols}, index=final_df1.index)
for col in cat_cols:
    final_df1[col] = df[col]


# In[53]:


# final_df.ncodpers = final_df.ncodpers.astype(int)
final_df1.head()


# ## MODELLING 
# 
# for this case study i have divided training data only into Train/ test dataset. As the volumne of the data is really high so it was faster this way

# In[54]:


training_dates = ['2015-07-28','2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28',                '2016-02-28', '2016-03-28', '2016-04-28']
test_dates = '2016-05-28'

final_df_training = final_df1[final_df1['fecha_dato'] != test_dates ]
final_df_testing = final_df1[final_df1['fecha_dato'] == test_dates ]


# In[55]:


(final_df_training_x,final_df_training_y) = (final_df_training[trainingcols], final_df_training[label_cols])
(final_df_test_x,final_df_test_y) = (final_df_testing[trainingcols], final_df_testing[label_cols])


# In[56]:


print final_df_training_x.columns
print final_df_training_y.columns
print final_df_test_x.columns
print final_df_test_y.columns


# ### We will be using XGBoost Model for this case study, We have other options as well. but, keeping time as a constraint 

# In[57]:


import warnings
warnings.filterwarnings("ignore")

xgtrain = xgb.DMatrix(final_df_training_x, label = final_df_training_y)
xgtest = xgb.DMatrix(final_df_test_x, label = final_df_test_y)
watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 


# In[ ]:


# We will be using Encoder in future to on hot encode some of the columns
'''
def encoderdf(df, feature):
    dummies = pd.get_dummies(df[feature])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature], axis=1)
    return(res)

encoderdf(final_df, cat_cols )'''


# In[58]:


## Define the parameters for the model

random_state = 4
params = {
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'alpha': 0,
        'lambda': 1,
        'verbose_eval': True,
        'seed': random_state,
        'num_class': 24,
        'objective': "multi:softprob",
        'eval_metric': 'mlogloss'
    }


# ### Train the model

# In[59]:



iterations = 40
printN = 1
#early_stopping_rounds = 10

xgbModel = xgb.train(params, 
                      xgtrain, 
                      iterations, 
                      watchlist,
                      verbose_eval = printN
                      #early_stopping_rounds=early_stopping_rounds
                      )


# In[61]:


import operator
importance = xgbModel.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print importance


# ### Do the prediction for the test data

# In[62]:


xgbtest = xgb.DMatrix(final_df_test_x)
XGBpreds = xgbModel.predict(xgbtest)


# In[63]:


XGBpreds


# In[64]:


pred = np.argsort(XGBpreds, axis=1)
pred = np.fliplr(pred) 


# In[66]:


test_ids = np.array(final_df_test_x['ncodpers'])
target_cols = np.array(target_cols)
f_preds = []

#iterate through our model's predictions (pred) and add the 7 most recommended products that the customer does not have.

for idx,predicted in enumerate(pred):
    ids = test_ids[idx]
    top_product = target_cols[predicted]
    new_top_product = []
    for product in top_product:
        new_top_product.append(product)
        if len(new_top_product) == 7:
            break
    f_preds.append(' '.join(new_top_product))


# ## FINAL OUTPUT

# In[67]:


res = pd.DataFrame({'ncodpers':test_ids,'added_products':f_preds})


# In[69]:


res[0:10]


# In[81]:


# Next Stpe is to cacluate MAP (mean average precision)
# Will be including that in future exercise

pred[4][0]


# In[80]:


test_ids[4]


# In[82]:


import numpy as np
from sklearn.metrics import average_precision_score


# In[99]:


prediction_1 = []
for i in range(len(pred)):
    prediction_1.append(pred[i][0])


# In[94]:


actuals = []
for row,index in final_df_test_y.iterrows():
    actuals.append(row)


# In[101]:


prediction_1 = np.array(prediction_1)
actuals = np.array(prediction_1)

