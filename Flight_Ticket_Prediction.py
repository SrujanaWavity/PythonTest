#!/usr/bin/env python
# coding: utf-8

# # PROJECT - 1
# 
# # Ticket Price Prediction                                                            

# ## General steps to be Followed in this project are:
# ### 1. Business Problem
# ### 2. Data Collection and Importing Data
# ### 3. Data Cleaning and processing
# ### 4. Exploring the Data
# ### 5. EDA
# ### 6. Feature Engineering 
# ### 7. ML
# ### 8. Conclusion

# In[15]:


# importing lybraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


# import CSV file
data = pd.read_csv(r'C:\Users\spand\Downloads\Data_Train.csv')
data.head()


# In[17]:


data.shape


# In[18]:


data.info()


# In[19]:


data.isnull().sum()


# In[20]:


# doing statistical analysis
data.describe()


# In[21]:


# Fetching those rows where null value is present
data[(data.Route.isna() ) |(data.Total_Stops.isna()) ]


# In[22]:


data.iloc[9037:9043]


# In[23]:


data.dropna(inplace=True, axis=0)


# # Exploratory Data Analysis

# ## Feature Engineering
# ### Duration
# converting duration into minutes_duration

# In[24]:


def Actual_Duration(Duration):
    if len(Duration.split()) == 2:
        hours = int(Duration.split()[0][:-1] )
        minutes = int(Duration.split()[1][:-1])
        return (hours*60)+ minutes
    else:
        return int(Duration[:-1])*60


# In[25]:


data['Duration'] = data['Duration'].apply(Actual_Duration)
data


# ### Arrival_Time and Dep_Time

# In[26]:


# changing Dep_time into datetime datatype
data.Dep_Time = pd.to_datetime(data.Dep_Time)
# Changing Arrival_Time into datetime datatype
data.Arrival_Time =pd.to_datetime(data.Arrival_Time)


# In[27]:


data['Dep_time_hours'] = data.Dep_Time.dt.hour
data['Dep_time_minutes'] = data.Dep_Time.dt.minute
data['Arrival_time_hours'] = data.Arrival_Time.dt.hour
data['Arrival_time_minutes'] = data.Arrival_Time.dt.minute
data.head(3)


# In[28]:


# dropping Dep_Time and Arrival_Time
data.drop(['Dep_Time','Arrival_Time'],axis = 1, inplace = True)
data.head(3)


# ### Date of Journey

# In[29]:


# changing date_of_journey to datetime
data.Date_of_Journey = pd.to_datetime(data.Date_of_Journey)
data.head(2)


# In[30]:


data['Month'] = data.Date_of_Journey.dt.month
data['Day']  = data.Date_of_Journey.dt.day
data.head(2)


# In[31]:


# dropping Date_of_Jourey as we extracted day and month but year is only 2019 so we will delete this column 
data.drop(['Date_of_Journey'], inplace= True, axis = 1)
data.head(2)


# ### Total Stops

# In[32]:


data[data.Total_Stops.isna()]
data.dropna(inplace = True, axis = 0)


# In[33]:


data[data.Total_Stops.isna()]


# In[34]:


# knowing the unique values in Total stops
data.Total_Stops.unique()


# In[35]:


data.Total_Stops = data.Total_Stops.map({'non-stop':0,
                     '1 stop':1,
                     '2 stops':2,
                     '3 stops':3,
                     '4 stops':4})


# ### Additional Information

# In[36]:


data.Additional_Info.value_counts()


# In[38]:


m_val = ((data.Additional_Info == 'No info').sum()+3)/ len(data)* 100
print(m_val)


# our feature Additional information containing more than 78% values as no information I am dropping this column 

# In[39]:


data.drop(['Additional_Info'],inplace =True, axis=1)


# In[40]:


data.head(2)


# In[41]:


data.select_dtypes(['object']).columns


# In[24]:


sns.pairplot(data, hue = 'Month')
plt.show()


# ### Airline

# In[43]:


data.Airline.value_counts()


# In[44]:



sns.boxplot(x= 'Airline',y='Price',data = data)
plt.xticks(rotation=90)
plt.title('BOX PLOT (Outliers)',color = 'Green')
plt.show()


# In[45]:


Airline_dummies = pd.get_dummies(data['Airline'],drop_first=True)


# In[46]:


Airline_dummies.head(2)


# In[47]:


data = pd.concat([data,Airline_dummies],axis = 1)
data.head(2)


# In[48]:


# dropping Airline column
data.drop(['Airline'], axis = 1,inplace = True)
data.head(2)


# In[49]:


secon = ['Source','Destination']
data = pd.get_dummies(data= data, columns = secon, drop_first = True)
data.head(2)


# ### Route

# In[50]:


Route = data[['Route']]
Route.head(2)


# In[51]:


Route['Route_1'] = Route['Route'].str.split('?').str[0]
Route['Route_2'] = Route['Route'].str.split('?').str[1]
Route['Route_3'] = Route['Route'].str.split('?').str[2]
Route['Route_4'] = Route['Route'].str.split('?').str[3]
Route['Route_5'] = Route['Route'].str.split('?').str[4]
Route.head(2)


# In[52]:


Route.fillna('None',inplace = True)
Route.head(2)


# In[53]:


# applying Label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(1,6):
    col = 'Route_' +str(i)
    Route[col] = le.fit_transform(Route[col])

Route.head(2)    


# In[54]:


Route.drop('Route',inplace = True, axis=1)
Route.head(2)


# In[55]:


data = pd.concat([data,Route],axis= 1)
data.head(2)


# In[60]:


data.drop('Route',axis=1 ,inplace=True)


# In[62]:


column = data.columns.to_list()
new_columns = column[:2]+ column[3:]
new_columns.append(column[2])
data = data.reindex(columns= new_columns)
data.head()


# In[65]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data = scaler.fit_transform(data)


# In[75]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score


# In[67]:


X = data[:,:-1]
y = data[:,1]


# In[68]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 10)


# In[69]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[80]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, min_samples_split=3)
model.fit(x_train,y_train)


# In[82]:


pd = model.predict(x_test)


# In[93]:


import numpy as np
from sklearn.metrics import mean_squared_error
def m(y_test,pd):
    print('rmsc', mean_squared_error(y_test,pd)**0.5)
    print('r2',r2_score(y_test,pd))
    
def a(y_test,pd):
    error = abs(y_test - pd)
    mapping = 100*np.mean(error/pd)
    accuracy =100- mapping
    return accuracy


# In[94]:


m(y_test,pd)


# In[95]:


a(y_test,pd)


# In[ ]:




