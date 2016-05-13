
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')


# In[2]:

from sie import *
from sie.mcmc import *


# ## Fit a coin flip model

# In[3]:

h,N=data=17,25


# In[4]:

def P_data(data,theta):
    h,N=data
    distribution=Bernoulli(h,N)
    return distribution(theta)


# In[5]:

model=MCMCModel(data,P_data,
                theta=Uniform(0,1))


# In[6]:

model.run_mcmc(500)
model.plot_chains()


# ### run a bit longer...

# In[7]:

model.run_mcmc(500)
model.plot_chains()


# ### Plot the MCMC distribution of $\theta$ and the Beta distribution solution
# 
# Hint: they should be the same

# In[8]:

model.plot_distributions()

dist=coinflip(h,N)
x=linspace(0,1.0,100)
px=dist.pdf(x)
plot(x,px,'r-')


# ### Look at some probabilitiess

# In[9]:

model.P('theta>0.5')


# In[10]:

model.P('(0.2<theta) & (theta<.5)')


# ## Regression Example

# In[78]:

N=1000
x=arange(N)/1000.0
y=randn(N)+40+.25*x
plot(x,y,'o')


# In[79]:

def constant(x,a):
    return a

model=MCMCModel_Regression(x,y,constant,
            a=Uniform(0,100),
            )
model.run_mcmc(500)
model.plot_chains()


# In[80]:

model.plot_distributions()


# In[81]:

plot(x,y,'o')

xfit=linspace(min(x),max(x),200)
yfit=model.predict(xfit)

plot(xfit,yfit,'-')


# In[82]:

plot(x,y,'o')
model.plot_predictions(xfit,color='g')


# ## Linear Model

# In[83]:

def linear(x,a,b):
    return a*x+b

model=MCMCModel_Regression(x,y,linear,
                a=Uniform(-10,10),
                b=Uniform(0,100),
                )

model.run_mcmc(500)
model.plot_chains()


# In[84]:

plot(x,y,'o')
model.plot_predictions(xfit,color='g')


# In[85]:

model.plot_distributions()


# In[86]:

model.triangle_plot()


# In[87]:

model.percentiles([5,50,95])


# In[88]:

model.P('a>0')


# In[ ]:



