
# coding: utf-8

# In[1]:

from sie import *


# ## Estimating Lengths
# 
# ### Known deviation, $\sigma$

# In[2]:

x=[5.1, 4.9, 4.7, 4.9, 5.0]
sigma=0.5


# In[7]:

mu=sample_mean(x)
N=len(x)


# In[8]:

dist=normal(mu,sigma/sqrt(N))
distplot(dist)


# In[9]:

credible_interval(dist)


# ### Unknown $\sigma$

# In[10]:

mu=sample_mean(x)
s=sample_deviation(x)
print mu,s


# In[26]:

dist=tdist(N-1,mu,s/sqrt(N))


# In[30]:

distplot(dist,xlim=[4.6,5.4])


# In[31]:

credible_interval(dist)


# In[ ]:



