
# coding: utf-8

# In[1]:

from sie import *


# ##Shoe size data

# In[2]:

data=load_data('data/shoesize.xls')


# In[3]:

data.head()


# ### Get a subset

# In[4]:

import random


# In[5]:

random.seed(102)
rows = random.sample(data.index, 10)
newdata=data.ix[rows]
data=newdata
data


# In[7]:

plot(data['Height'],data['Size'],'o')
gca().set_xlim([60,72])
gca().set_ylim([4,14])
xlabel('Height [inches]')
ylabel('Shoe Size')


# ### Do the regression

# In[8]:

result=regression('Size ~ Height',data)


# In[10]:

plot(data['Height'],data['Size'],'o')

h=linspace(60,72,10)
plot(h,result['_Predict'](Height=h),'-')

gca().set_xlim([60,72])
gca().set_ylim([4,14])
xlabel('Height [inches]')
ylabel('Shoe Size')

b=result.Intercept.mean()
m=result.Height.mean()

if b>0:
    text(62,12,'$y=%.3f x + %.3f$' % (m,b),fontsize=30)
else:
    text(62,12,'$y=%.3f x %.3f$' % (m,b),fontsize=30)


# ## SAT Data

# In[12]:

data=load_data('data/sat.csv')


# ### Simple Linear Regression

# In[13]:

result=regression('total ~ expenditure',data)


# In[15]:

plot(data['expenditure'],data['total'],'o')
xlabel('Expenditure [per pupil, thousands]')
ylabel('SAT Total')
h=linspace(3,10,10)
plot(h,result['_Predict'](expenditure=h),'-')

b=result.Intercept.mean()
m=result.expenditure.mean()

if b>0:
    text(4.5,1125,'$y=%.3f x + %.3f$' % (m,b),fontsize=30)
else:
    text(4.5,1125,'$y=%.3f x %.3f$' % (m,b),fontsize=30)


# In[16]:

result=regression('percent_taking ~ expenditure',data)


# In[19]:

plot(data['expenditure'],data['percent_taking'],'o')
xlabel('Expenditure [per pupil, thousands]')
ylabel('SAT Total')
h=linspace(3,10,10)
plot(h,result['_Predict'](expenditure=h),'-')

b=result.Intercept.mean()
m=result.expenditure.mean()

if b>0:
    text(4.5,85,'$y=%.3f x + %.3f$' % (m,b),fontsize=30)
else:
    text(4.5,85,'$y=%.3f x %.3f$' % (m,b),fontsize=30)


# ### Multiple Regression

# In[20]:

result=regression('total ~ expenditure + percent_taking',data)


# In[ ]:



