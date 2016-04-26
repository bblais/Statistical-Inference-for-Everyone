
# coding: utf-8

# In[1]:

from sie import *


# ## Iris Example

# In[2]:

data=load_data('data/iris.csv')


# In[3]:

x_sertosa=data[data['class']=='Iris-setosa']['petal length [cm]']


# In[4]:

x=x_sertosa
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_sertosa=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma


# ### $z$-test - is a new point the same type of flower?

# In[11]:

new_length=1.7


# In[21]:

distplot(t_sertosa,label='petal length',xlim=[1.37,1.8],
                 quartiles=[.01,0.05,.5,.95,.99],
)
ax=gca()
ax.axvline(1.7,color='r')
savefig('../../figs/z_test_iris.pdf')


# In[ ]:



