#!/usr/bin/env python
# coding: utf-8

# <font color=red>Import Pandas

# In[1]:


import pandas as pd


# <font color=red>Import 'house_rental_data.csv.txt'

# In[2]:


house_rental_data = pd.read_csv('house_rental_data.csv.txt')
house_rental_data


# <font color=red>Remove 'Unnamed' column

# In[8]:


house_rental_data = house_rental_data.drop(house_rental_data.columns[0], axis = 1)
house_rental_data


# <font color=red>Find 'Number of rows and columns'

# In[9]:


house_rental_data.shape


# <font color=red>Data Preparation and Train-Test Split for Machine Learning

# Data Splitting: Feature and Target Variable Separation
# <br>Drop 'Price' Column and save as variable x and y
# <br><font color=green>Press only one time shift+Enter, function will run otherwise it will through error if we press more that one.

# In[10]:


x = house_rental_data.drop(['Price'], axis=1)
y = house_rental_data['Price']


# Splitting Data into Training and Testing Sets using scikit-learn's train_test_split
# <br><font color=green>Note: The test_size parameter is set to 0.2, which means that 20% of the data will be allocated to the testing set, while the remaining 80% will be used for training. 
# <br>The random_state parameter is set to 0 to ensure reproducibility of the split.

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


# Find 'Number of rows' and 'Number of columns' x_test

# In[12]:


x_test.shape


# Training a K-Nearest Neighbors Regressor Model

# In[13]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)


# Generating Predictions using the K-Nearest Neighbors Regressor Model

# In[14]:


y_pred = knr.predict(x_test)
y_pred


# Making Single Data Point Prediction using the K-Nearest Neighbors Regressor Model 
# <br>Also import numpy

# In[27]:


import numpy as np
input_data = (2,1,3,5,4)
convert_to_array = np.asarray(input_data)
re_shape = convert_to_array.reshape(1,-1)
prediction = knr.predict(re_shape)
print(prediction)


# Assessing the Accuracy of the K-Nearest Neighbors Regressor Model with the score() Function

# In[25]:


knr.score(x_test, y_test)

