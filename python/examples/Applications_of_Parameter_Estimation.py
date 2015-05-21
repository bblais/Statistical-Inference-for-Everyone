# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sie import *

# <markdowncell>

# ## Iris Example

# <codecell>

data=load_data('data/iris.csv')

# <codecell>

x_sertosa=data[data['class']=='Iris-setosa']['petal length [cm]']
x_virginica=data[data['class']=='Iris-virginica']['petal length [cm]']
x_versicolor=data[data['class']=='Iris-versicolor']['petal length [cm]']

# <codecell>

print x_sertosa[:10]  # print the first 10

# <codecell>

x=x_sertosa
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_sertosa=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma

# <codecell>

x=x_versicolor
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_versicolor=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma

# <codecell>

x=x_virginica
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_virginica=tdist(N,mu,sigma)

print "total number of data points:",N
print "best estimate:",mu
print "uncertainty:",sigma

# <codecell>

distplot2([t_sertosa,t_versicolor,t_virginica],show_quartiles=False)

# <codecell>

distplot(t_virginica)

# <codecell>

credible_interval(t_versicolor)

# <codecell>

credible_interval(t_virginica)

# <markdowncell>

# ## Sunrise

# <codecell>

dist=beta(h=365,N=365)

# <codecell>

distplot(dist)

# <codecell>

credible_interval(dist)

# <markdowncell>

# ## Cancer Example

# <codecell>

dist=beta(h=7,N=10)

# <codecell>

distplot(dist,figsize=(8,5))

# <codecell>

credible_interval(dist)

# <markdowncell>

# Essentially no evidence of any effect over 50 percent.

# <markdowncell>

# ## Pennies

# <codecell>

data1=load_data('data/pennies1.csv')
print data1
year,mass=data1['Year'],data1['Mass [g]']

# <codecell>

plot(year,mass,'o')
xlabel('year')
ylabel('Mass per Penny [g]')

# <codecell>

x=mass
mu=sample_mean(x)
N=len(x)
sigma=sample_deviation(x)/sqrt(N)
t_penny1=tdist(N,mu,sigma)

distplot(t_penny1,label='mass [g]')

# <codecell>

CI=credible_interval(t_penny1,percentage=99)
print CI

# <codecell>

plot(year,mass,'o')
credible_interval_plot(t_penny1,percentage=99)
xlabel('year')
ylabel('Mass per Penny [g]')

# <markdowncell>

# ### Do the 2 datasets

# <codecell>

data2=load_data('data/pennies2.csv')
print data2
year1,mass1=year,mass
year2,mass2=data2['Year'],data2['Mass [g]']

# <codecell>

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

# <codecell>

plot(year1,mass1,'o')
credible_interval_plot(t_penny1,percentage=99)
plot(year2,mass2,'ro')
credible_interval_plot(t_penny2,percentage=99,xlim=[1989,2005])
xlabel('year')
ylabel('Mass per Penny [g]')

# <markdowncell>

# ### Distribution of the difference, normal approximation

# <codecell>

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

# <markdowncell>

# clearly larger than zero at well over the 99% level.

# <markdowncell>

# ## Ball Bearing Sizes

# <codecell>

data1=[1.18,1.42,0.69,0.88,1.62,1.09,1.53,1.02,1.19,1.32]
data2=[1.72,1.62,1.69,0.79,1.79,0.77,1.44,1.29,1.96,0.99]
N1=len(data1)
N2=len(data2)

# <codecell>

mu1=sample_mean(data1)
mu2=sample_mean(data2)
print mu1,mu2

# <codecell>

S1=sample_deviation(data1)
S2=sample_deviation(data2)
print S1,S2

# <codecell>

sigma1=S1/sqrt(N1)
sigma2=S2/sqrt(N2)
print sigma1,sigma2

# <codecell>

dist1=normal(mu1,sigma1)
dist2=normal(mu2,sigma2)
distplot2([dist1,dist2],show_quartiles=False,label='size [microns]')
legend([r'$\mu_1$',r'$\mu_2$'])

# <codecell>


