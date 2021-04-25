#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Import dataset and extract dependent and independent variables

# In[3]:


#Import dataset and extract dependent and independent variables
salary_data = pd.read_csv("Salary_Data.csv")

# Salary Find what will be the salary the person
x = salary_data.iloc[:, :-1].values
y = salary_data.iloc[:, -1].values


# In[4]:


x


# In[5]:


y


# In[6]:


salary_data


# ### Visualising the dataset

# In[9]:


sns.distplot(salary_data['YearsExperience'],kde = False,bins = 10)


# In[27]:


sns.countplot(y='YearsExperience',data = salary_data)


# In[28]:


sns.barplot(x='YearsExperience',y='Salary',data = salary_data)


# In[29]:


sns.heatmap(salary_data.corr())


# ### Splitting the dataset into the Training and Test Set

# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# ### Fitting Simple linear regression to the training set

# In[32]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# ### Predicting the test set result

# In[35]:


y_pred = lr.predict(x_test)
y_pred


# ### Visualising the training set result

# In[39]:


plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train,lr.predict(x_train), color = 'red')
plt.title('Salary ~ Experiance (Train set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()


# ### Visualising the test set result

# In[40]:


plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train,lr.predict(x_train), color = 'red')
plt.title('Salary vs Experiance (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('salary')
plt.show()


# ### Finding the residuals

# In[43]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))


# In[ ]:




