
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#data filtering steps from EDA jupyter notebook
items = pd.read_csv('data.csv')
items_upd = items[items['weekly_sales'].notna()]
items_upd = items_upd[items_upd['number_of_item_on_file']<500]
items_upd=items_upd[(items_upd['weekly_sales']>=0.000000000000000001) & (items_upd['weekly_sales']<1000)]
items_upd['date'] = pd.to_datetime(items_upd['date'])

#data
print('Shape of dataset is:',items_upd.shape)
print('\n',items_upd.head())

#adding features for weekday
items_upd['date'].dt.day_name().value_counts()
items_upd['dayofweek']=items_upd['date'].dt.day_name()

#creating dummy variables for all categorical columns
#dayofweek, item_category, store_type
items_final = pd.get_dummies(items_upd,columns=['dayofweek','item_category','store_type'],
                             drop_first=True)




#creating X and y to feed to models

y=items_final['in_store_flag']

items_final.columns
X=pd.concat([items_final.iloc[:,3:6],items_final.iloc[:,7:]],axis=1)


### train test split
seed=0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
                                                    
print('\n Shape of X train dataset is:',X_train.shape)
print('\n Shape of X test dataset is:',X_test.shape)