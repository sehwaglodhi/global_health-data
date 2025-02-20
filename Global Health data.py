#!/usr/bin/env python
# coding: utf-8

# # Imputation and encoding of data
# 

# In[189]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[190]:


#importing dataset
dataset=pd.read_csv("E:\world_health_data.csv")


# In[191]:


dataset.head(20)


# In[192]:


X=dataset.drop(["health_exp","maternal_mortality","inci_tuberc","prev_undernourishment","life_expect"],axis=1)
Y=dataset["life_expect"]


# In[193]:


print(X)


# In[194]:


print(Y)


# In[195]:


from sklearn.impute import SimpleImputer
Numeric_cols=X.select_dtypes(include=['int64','float64']).columns
Object_cols=X.select_dtypes(include=['object']).columns

#fit in the simpleImputer
Numeric_Impu=SimpleImputer(strategy='mean')
X[Numeric_cols]=Numeric_Impu.fit_transform(X[Numeric_cols])

Object_Imputer=SimpleImputer(strategy='most_frequent')
X[Object_cols]=Object_Imputer.fit_transform(X[Object_cols])

#handling missing data in y
#impute function take datframe so i have to convert it into the series to dataframe
Y=Y.to_frame()
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
Y=imp.fit_transform(Y) 
Y=pd.Series(Y.flatten())
                                              



# In[196]:


print(X)


# In[197]:


#  Step 4: Encode categorical data
# Option 1: OneHotEncoder for categorical features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[Object_cols]), columns=encoder.get_feature_names_out(Object_cols), index=X.index)

# Drop original categorical columns and replace with encoded columns
X = pd.concat([X.drop(columns=Object_cols), X_encoded], axis=1)



# In[198]:


print(X)


# In[199]:


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# clt = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# X=np.array(clt.fit_transform(X))


# In[200]:


print(X)


# In[201]:


from sklearn.model_selection import train_test_split


# In[202]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=8)
print(X_train)


# In[208]:


from sklearn.linear_model import LinearRegression
print(Y_train)


# In[204]:


model=LinearRegression()


# In[205]:


model.fit(X_train,Y_train)


# In[206]:


Y_pred=model.predict(X_test)


# In[207]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_test,Y_pred,color='blue')


# In[ ]:




