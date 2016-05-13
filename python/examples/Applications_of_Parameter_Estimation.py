
# coding: utf-8

# In[1]:

from sie import *


# ## Iris Example

# In[33]:

data=load_data('data/iris.csv')


# In[34]:

x_sertosa=data[data['class']=='Iris-setosa']['petal length [cm]']
x_virginica=data[data['class']=='Iris-virginica']['petal length [cm]']
x_versicolor=data[data['class']=='Iris-versicolor']['petal length [cm]']


# In[40]:

print x_sertosa[:10]  # print the first 10


# In[41]:

x=x_sertosa
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_sertosa=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma


# In[42]:

x=x_versicolor
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_versicolor=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma


# In[43]:

x=x_virginica
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_virginica=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma


# In[9]:

distplot(t_virginica)


# In[8]:

distplot2([t_sertosa,t_versicolor,t_virginica],show_quartiles=False)


# In[10]:

credible_interval(t_versicolor)


# In[11]:

credible_interval(t_virginica)


# ## Sunrise

# In[ ]:

from sie import *


# If you knew nothing about sunrises, and watched a year of them, what is the probability of another one tomorrow?

# In[12]:

dist=beta(h=365,N=365)


# In[13]:

distplot(dist)


# In[14]:

credible_interval(dist)


# ## Cancer Example

# In[15]:

dist=beta(h=7,N=10)


# In[16]:

distplot(dist,figsize=(8,5))


# In[17]:

credible_interval(dist)


# Essentially no evidence of any effect over 50 percent.

# ## Pennies

# In[5]:

data1=load_data('data/pennies1.csv')
print data1
year,mass=data1['Year'],data1['Mass [g]']


# In[6]:

plot(year,mass,'o')
xlabel('year')
ylabel('Mass per Penny [g]')


# In[7]:

x=mass
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_penny1=tdist(N,mu,sigma)

distplot(t_penny1,label='mass [g]')


# In[8]:

CI=credible_interval(t_penny1,percentage=99)
print CI


# In[9]:

plot(year,mass,'o')
credible_interval_plot(t_penny1,percentage=99)
xlabel('year')
ylabel('Mass per Penny [g]')


# ### Do the 2 datasets

# In[10]:

data2=load_data('data/pennies2.csv')
print data2
year1,mass1=year,mass
year2,mass2=data2['Year'],data2['Mass [g]']


# In[11]:

x=mass1
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_penny1=tdist(N,mu,sigma)

x=mass2
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_penny2=tdist(N,mu,sigma)

distplot2([t_penny1,t_penny2],show_quartiles=False,label='mass [g]')
legend([r'$\mu_1$',r'$\mu_2$'])


# In[12]:

plot(year1,mass1,'o')
credible_interval_plot(t_penny1,percentage=99)
plot(year2,mass2,'ro')
credible_interval_plot(t_penny2,percentage=99,xlim=[1989,2005])
xlabel('year')
ylabel('Mass per Penny [g]')


# ### Distribution of the difference, normal approximation

# In[13]:

N1=len(mass1)
N2=len(mass2)

mu1=sample_mean(mass1)
mu2=sample_mean(mass2)


sigma1=(1+20.0/N1**2)*sample_deviation(mass1)/sqrt(N1)
sigma2=(1+20.0/N2**2)*sample_deviation(mass2)/sqrt(N1)


delta_12=mu1-mu2
sigma_delta12=sqrt(sigma1**2+sigma2**2)

dist_delta=normal(delta_12,sigma_delta12)
distplot(dist_delta)


# clearly larger than zero at well over the 99% level.

# ## Ball Bearing Sizes

# In[23]:

data1=[1.18,1.42,0.69,0.88,1.62,1.09,1.53,1.02,1.19,1.32]
data2=[1.72,1.62,1.69,0.79,1.79,0.77,1.44,1.29,1.96,0.99]
N1=len(data1)
N2=len(data2)


# In[2]:

mu1=sample_mean(data1)
mu2=sample_mean(data2)
print mu1,mu2


# In[24]:

S1=sample_deviation(data1)
S2=sample_deviation(data2)
print S1,S2


# In[25]:

sigma1=S1/sqrt(N1)
sigma2=S2/sqrt(N2)
print sigma1,sigma2


# In[30]:

dist1=normal(mu1,sigma1)
dist2=normal(mu2,sigma2)
distplot2([dist1,dist2],show_quartiles=False,label='size [microns]')
legend([r'$\mu_1$',r'$\mu_2$'])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



