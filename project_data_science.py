#!/usr/bin/env python
# coding: utf-8

# # Topic Name: Bank Customer Churn Model

# # Objective

# A Bank wants to take care of customer retention for its product: savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank.

# # Import Library

# In[10]:


import pandas as pd


# In[11]:


import numpy as np


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


import seaborn as sns


# # Import Dataset

# In[14]:


df = pd.read_csv('Bank.csv')


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.duplicated('CustomerId').sum()


# In[18]:


df = df.set_index('CustomerId')


# In[19]:


df.info()


# 
# # Encoding

# In[20]:


df['Geography'].value_counts()


# In[21]:


df.replace({'Geography':{'France': 2, 'Germany':1, 'Spain':0}}, inplace=True)


# In[22]:


df['Gender'].value_counts()


# In[23]:


df.replace({'Gender':{'Male':0,'Female' : 1}}, inplace=True)


# In[24]:


df['Num Of Products'].value_counts()


# In[25]:


df.replace({'Num Of Products': {1:0, 2:1, 3:1, 4:1}}, inplace=True)


# In[26]:


df['Has Credit Card'].value_counts()


# In[27]:


df['Is Active Member'].value_counts()


# In[28]:


df.loc[(df['Balance']==0), 'Churn'].value_counts()


# In[29]:


df['Zero Balace'] = np.where(df['Balance']>0, 1, 0)


# In[30]:


df['Zero Balace'].hist()


# In[31]:


df.groupby(['Churn', 'Geography']).count()


# # Define Label and Features

# In[32]:


df.columns


# In[33]:


x = df.drop(['Surname', 'Churn'], axis = 1)


# In[34]:


y = df['Churn']


# In[35]:


x.shape, y.shape


# # Handling Imbalance Data

# Class imbalace is a common problem in Machine learning, especially in classification problems as machine learning algorithms are designed to maximize accuracy and reduce errors.
# 

# To Overcome this Imbalance of data , we are using two Strategies:
#    1. Undersampling : Undersampling can be defined as removing some observation of the majority class.This is done until the majority and minority class is balanced out.
#     
#    2. Oversampling : Oversampling can be defined as adding more copies to the minority class.Oversampling can be a good choise when you don't have a ton of data to work with. 

# In[36]:


df['Churn'].value_counts()


# In[37]:


sns.countplot(x ='Churn', data = df);


# In[38]:


x.shape, y.shape


# # Random Under Sampling

# In[40]:


from imblearn.under_sampling import RandomUnderSampler


# In[41]:


rus = RandomUnderSampler(random_state=2529)


# In[42]:


x_rus, y_rus = rus.fit_resample(x, y)


# In[43]:


x_rus.shape, y_rus.shape, x.shape, y.shape


# In[44]:


y.value_counts()


# In[45]:


y_rus.value_counts()


# In[46]:


y_rus.plot(kind = 'hist')


# # Random Over Sampling

# In[47]:


from imblearn.over_sampling import RandomOverSampler


# In[48]:


ros = RandomOverSampler(random_state=2529)


# In[49]:


x_ros, y_ros = ros.fit_resample(x, y)


# In[50]:


x_ros.shape, y_ros.shape, x.shape, y.shape


# In[51]:


y.value_counts()


# In[52]:


y_ros.value_counts()


# In[53]:


y_ros.plot(kind = 'hist')


# # Train Test Split

# In[54]:


from sklearn.model_selection import train_test_split


# Split Original Data

# In[55]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=2529)


# Split Random Under Sample Data

# In[56]:


x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus,test_size=0.3, random_state=2529)


# Split Random Over Sample Data

# In[57]:


x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros,test_size=0.3, random_state=2529)


# # Standardize Features

# In[58]:


from sklearn.preprocessing import StandardScaler


# In[59]:


sc = StandardScaler()


# Standardize Original Data

# In[60]:


x_train[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_train[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# In[61]:


x_test[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_test[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# Standardize Random Under Sample Data

# In[62]:


x_train_rus[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_train_rus[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# In[63]:


x_test_rus[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_test_rus[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# Standardize Random Over Sample Data

# In[64]:


x_train_ros[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_train_ros[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# In[65]:


x_test_ros[['CreditScore','Age','Tenure','Balance','Estimated Salary']] = sc.fit_transform(x_test_ros[['CreditScore','Age','Tenure','Balance','Estimated Salary']])


# # Support Vector Machine Classifier

# In[66]:


from sklearn.svm import SVC


# In[67]:


svc = SVC()


# In[68]:


svc.fit(x_train, y_train)


# In[69]:


y_pred = svc.predict(x_test)


# # Model Accuracy

# In[70]:


from sklearn.metrics import confusion_matrix, classification_report


# In[71]:


confusion_matrix(y_test, y_pred)


# In[72]:


print(classification_report(y_test, y_pred))


# #   Hyperparameter Tunning

# In[73]:


from sklearn.model_selection import GridSearchCV


# In[74]:


param_grid = {'C' : [0.1,1,10],
             'gamma' : [1,0.1,0.01],
             'kernel' : ['rbf'],
             'class_weight' : ['balanced']}


# In[75]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 2)
grid.fit(x_train,y_train)


# In[77]:


print(grid.best_estimator_)


# In[80]:


grid_predictions = grid.predict(x_test)


# In[83]:


confusion_matrix(y_test,grid_predictions)


# In[85]:


print(classification_report(y_test,grid_predictions))


# In[87]:


svc_rus =SVC()


# In[89]:


svc_rus.fit(x_train_rus,y_train_rus)


# In[91]:


y_pred_rus = svc_rus.predict(x_test_rus)


# # Model Accuracy

# In[93]:


confusion_matrix(y_test_rus, y_pred_rus)


# In[95]:


print(classification_report(y_test_rus,y_pred_rus))


# # Hyperparameter Tunning

# In[97]:


param_grid = {'C' : [0.1,1,10],
             'gamma' : [1,0.1,0.01],
             'kernel' : ['rbf'],
             'class_weight' : ['balanced']}


# In[100]:


grid_rus = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 2)
grid_rus.fit(x_train_rus,y_train_rus)


# In[102]:


print(grid_rus.best_estimator_)


# In[103]:


grid_predictions_rus = grid_rus.predict(x_test_rus)


# In[110]:


confusion_matrix(y_test_rus,grid_predictions_rus)


# In[107]:


print(classification_report(y_test_rus,grid_predictions_rus))


# # Model with Random Over Sampling

# In[109]:


svc_ros = SVC()


# In[111]:


svc_ros.fit(x_train_ros, y_train_ros)


# In[112]:


y_pred_ros = svc_ros.predict(x_test_ros)


# # Model Accuracy

# In[114]:


confusion_matrix(y_test_ros,y_pred_ros)


# In[115]:


print(classification_report(y_test_ros, y_pred_ros))


# # Hyperparameter Tunning

# In[117]:


param_grid = {'C' : [0.1,1,10],
             'gamma' : [1,0.1,0.01],
             'kernel' : ['rbf'],
             'class_weight' : ['balanced']}


# In[119]:


grid_ros = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 2)
grid_ros.fit(x_train_ros,y_train_ros)


# In[121]:


print(grid_ros.best_estimator_)


# In[123]:


grid_predictions_ros = grid_ros.predict(x_test_ros)


# In[125]:


confusion_matrix(y_test_ros,grid_predictions_ros)


# In[127]:


print(classification_report(y_test_ros,grid_predictions_ros))


# # Lets Compare

# In[129]:


print(classification_report(y_test, y_pred))


# In[131]:


print(classification_report(y_test, grid_predictions))


# In[133]:


print(classification_report(y_test_rus, y_pred_rus))


# In[135]:


print(classification_report(y_test_ros, y_pred_ros))


# In[137]:


print(classification_report(y_test_ros, grid_predictions_ros))


# In[ ]:




