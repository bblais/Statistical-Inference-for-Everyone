
# coding: utf-8

# In[1]:

from sie import *


# ## Beta Distribution Example
# 
# ### 3 heads and 9 tails

# Plot a beta distribution with 3 heads and 9 tails...

# In[4]:

dist=beta(h=1,N=3)
distplot(dist,xlim=[0,1],show_quartiles=False)


# The median of this distribution...

# In[5]:

dist.median()


# the 95% credible interval, with the median in the middle,

# In[5]:

credible_interval(dist)


# ### 1 heads and 3 tails
# 
# This should be about the same fraction as the previous example, but broader

# In[16]:

dist=beta(h=1,N=4)
distplot(dist,xlim=[0,1])


# In[17]:

credible_interval(dist)


# In[ ]:



