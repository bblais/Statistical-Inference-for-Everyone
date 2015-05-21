
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
from sie import *
import sie


# In[3]:

sie.__file__


# In[4]:

data=pandas.read_csv('pennies/Penny Lab  - Sheet1.csv')


# In[5]:

data


# In[6]:

print data['Year'],data['Number of Pennies']


# In[7]:

year=data['Year'][:54]
N=data['Number of Pennies'][:54].astype(float)
total_mass=data['Mass (g)'][:54].astype(float)
total_height=data['Height (cm)'][:54].astype(float)
diameter=data['Diameter (cm)'][:54].astype(float)


# In[8]:

diameter


# In[9]:

figure(figsize=(16,9))
plot(year,N,'-o')
xlabel('year')
ylabel('Number of Pennies')
savefig('../../figs/number.pdf')


# In[10]:

total_volume=pi*(diameter/2.0)**2*total_height
volume_per_penny=total_volume/N
mass_per_penny=total_mass/N
density=total_mass/total_volume


# In[11]:

figure(figsize=(16,9))
plot(year,density,'-o')
xlabel('year')
ylabel('Density [g/cm$^3$]')
savefig('../../figs/density.pdf')


# In[12]:

figure(figsize=(16,9))
plot(year[year<1975],mass_per_penny[year<1975],'b-o',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
gca().set_ylim([3,3.2])
savefig('../../figs/mass1960_1974.pdf')

figure(figsize=(16,9))
plot(year[(1989<=year) & (year<2004)],mass_per_penny[(1989<=year) & (year<2004)],'b-o',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
savefig('../../figs/mass1989_2003.pdf')

figure(figsize=(16,9))
plot(year[year<1975],mass_per_penny[year<1975],'b-o')
plot(year[(1989<=year) & (year<2004)],mass_per_penny[(1989<=year) & (year<2004)],'b-o',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
savefig('../../figs/mass1965_2003.pdf')


# In[13]:

x=concatenate([array(year[year<1975])])
y=concatenate([array(mass_per_penny[year<1975])])

figure(figsize=(16,9))
plot(x,y,'b-o',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
#gca().set_xlim([1955,2005])
#gca().set_xticks(arange(1955,2010,5))

xl=gca().get_xlim()

N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

mu_p=mu+k*3*sigma
mu_m=mu-k*3*sigma

plot(xl,[mu,mu],'g--')
plot(xl,[mu_p,mu_p],'g:')
plot(xl,[mu_m,mu_m],'g:')


text(1973.5,3.15,r'Best estimate of "true" value: $\hat{\mu}=%.3f\pm%.4f$' % (mu,k*sigma),fontsize=30,ha='right')
text(1973.5,3.14,r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p),fontsize=30,ha='right')
savefig('../../figs/mass1965_1974_1_value.pdf')

print r"""
\begin{center}
\begin{tabular}{ccccc}
\toprule
{\bf Year} & {\bf Mass [g]} \\"""

for vx,vy in zip(x,y):
    print r"%d& %.3f\\" % (vx,vy)

print r"""\bottomrule
\end{tabular}
\end{center}
"""


# In[24]:

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)


dist=normal(mu,sigma)
distplot(dist,show_quartiles=False,xlim=[3,3.2],figsize=(16,9))
xlabel(r'$\mu$ [g]')
ylabel(r'$p(\mu)$')
text(3.01,52+10,r'Best Estimate for $\mu=%.3f \pm %.4f$ grams' % (mu,k*sigma),ha='left',fontsize=30) 
text(3.01,45+10,r'99%% CI for $\mu: [%.3f,%.3f]$' % (mu_m,mu_p),ha='left',fontsize=30) 
savefig('../../figs/mass1965_1974_1_value_dist.pdf')


# ### Two values

# In[46]:

x=concatenate([array(year[year<1975])])
y=concatenate([array(mass_per_penny[year<1975])])

figure(figsize=(16,9))
plot(x,y,'bo',markersize=10)

xl=[1960,1974]

N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

mu_p=mu+k*3*sigma
mu_m=mu-k*3*sigma


mu1=mu
sigma1=k*sigma


plot(xl,[mu,mu],'g--')
plot(xl,[mu_p,mu_p],'g:')
plot(xl,[mu_m,mu_m],'g:')


text(1985,2.96,r'Best estimate $\hat{\mu}_1=%.3f\pm%.4f$' % (mu,k*sigma),fontsize=30,ha='right')
text(1985,2.90,r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p),fontsize=30,ha='right')

print r'Best estimate $\hat{\mu}_1=%.3f\pm%.4f$' % (mu,k*sigma)
print r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p)

x=concatenate([array(year[(1989<=year) & (year<2004)])])
y=concatenate([array(mass_per_penny[(1989<=year) & (year<2004)])])

plot(x,y,'bo',markersize=10)

xl=[1988,2004]

N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

mu2=mu
sigma2=k*sigma

mu_p=mu+k*3*sigma
mu_m=mu-k*3*sigma

plot(xl,[mu,mu],'g--')
plot(xl,[mu_p,mu_p],'g:')
plot(xl,[mu_m,mu_m],'g:')


text(2002,2.66,r'Best estimate $\hat{\mu}_2=%.3f\pm%.4f$' % (mu,k*sigma),fontsize=30,ha='right')
text(2002,2.6,r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p),fontsize=30,ha='right')


xlabel('year')
ylabel('Mass per Penny [g]')

savefig('../../figs/mass1965_2003_2_value.pdf')


print r'Best estimate $\hat{\mu}_2=%.3f\pm%.4f$' % (mu,k*sigma)
print r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p)


# In[14]:

delta=mu1-mu2
sigma_12=sqrt(sigma1**2+sigma2**2)

dist=normal(delta,sigma_12)
distplot(dist,show_quartiles=False,xlim=[0,.7],figsize=(16,9))
xlabel(r'$\mu_1-\mu_2$')
ylabel(r'$p(\mu_1-\mu_2)$')
text(.6,52,r'Best Estimate for $\mu_1-\mu_2=%.3f \pm %.3f$' % (delta,sigma_12),ha='right',fontsize=30) 
text(.5,45,r'99%% CI for $\mu_1-\mu_2: [%.3f,%.3f]$' % (delta-3*sigma_12,delta+3*sigma_12),ha='right',fontsize=30) 
savefig('../../figs/mass1965_2003_2_value_diff.pdf')


# In[15]:

N


# In[16]:

S


# In[17]:

k


# In[18]:

k*sigma


# ### one value

# In[45]:

x=concatenate([array(year[year<1975]),year[(1989<=year) & (year<2004)]])
y=concatenate([array(mass_per_penny[year<1975]),mass_per_penny[(1989<=year) & (year<2004)]])

figure(figsize=(16,9))
plot(x,y,'bo',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
#gca().set_xlim([1955,2005])
#gca().set_xticks(arange(1955,2010,5))

xl=gca().get_xlim()

N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

mu_p=mu+k*3*sigma
mu_m=mu-k*3*sigma

plot(xl,[mu,mu],'g--')
plot(xl,[mu_p,mu_p],'g:')
plot(xl,[mu_m,mu_m],'g:')


text(2002,3.00,r'Best estimate of "true" value??: $\hat{\mu}=%.3f\pm%.4f$' % (mu,k*sigma),fontsize=30,ha='right')
text(2002,2.90,r'99%% CI: $[%.3f,%.3f]$' % (mu_m,mu_p),fontsize=30,ha='right')
savefig('../../figs/mass1965_2003_1_value.pdf')

print r"""
\begin{center}
\begin{tabular}{ccccc}
\toprule
{\bf Year} & {\bf Mass [g]} \\"""

for vx,vy in zip(x,y):
    print r"%d& %.3f\\" % (vx,vy)

print r"""\bottomrule
\end{tabular}
\end{center}
"""


# In[20]:

x=concatenate([array(year[year<1975]),year[(1989<=year) & (year<2004)]])
y=concatenate([array(mass_per_penny[year<1975]),mass_per_penny[(1989<=year) & (year<2004)]])

figure(figsize=(16,9))
plot(x,y,'bo',markersize=10)
xlabel('year')
ylabel('Mass per Penny [g]')
gca().set_xlim([1955,2005])
gca().set_xticks(arange(1955,2010,5))

xl=gca().get_xlim()

N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

plot(xl,[mu,mu],'g--')
plot(xl,[mu+k*3*sigma,mu+k*3*sigma],'g:')
plot(xl,[mu-k*3*sigma,mu-k*3*sigma],'g:')

savefig('../../figs/mass1965_2003_1_value.pdf')



# In[20]:




# In[20]:




# In[21]:

figure(figsize=(16,9))
plot(year,mass_per_penny,'-o')
xlabel('year')
ylabel('Mass per Penny [g]')
savefig('../../figs/mass.pdf')


# In[22]:

figure(figsize=(16,9))
plot(year,volume_per_penny,'-o')
xlabel('year')
ylabel('Volume per Penny [cm$^3$]')
savefig('../../figs/volume.pdf')


# ## Analysis

# In[23]:

result=fit(year,density,'linear')


# In[24]:

figure(figsize=(16,9))
plot(year,density,'-o')
density_fit=fitval(result,year)
plot(year,density_fit,'--')
p=result.params
text(1990,8.0,'y=%.3f x + %.3f ' % (p[0],p[1]),fontsize=30)
xlabel('year')
ylabel('Density [g/cm$^3$]')
savefig('../../figs/density_fit.pdf')


# In[25]:

x=array(year)
y=array(density)

x=x[isfinite(y)]
y=y[isfinite(y)]



N=len(x)
k=1+20.0/N**2

mu=mean(y)
S=std(y,ddof=1)
sigma=S/sqrt(N)

x=array(year)
y=array(density)


# In[26]:

figure(figsize=(16,9))
plot(x,y,'-o')
plot([1960,2014],[mu,mu],'g--')
plot([1960,2014],[mu+k*3*sigma,mu+k*3*sigma],'g:')
plot([1960,2014],[mu-k*3*sigma,mu-k*3*sigma],'g:')

text(1990,8.0,r'$\mu = %.4f \pm %.4f$' % (mu,k*sigma),fontsize=30)


xlabel('year')
ylabel('Density [g/cm$^3$]')
savefig('../../figs/density_fit2.pdf')


# In[27]:

x=array(year)
y=array(density)
x=x[isfinite(y)]
y=y[isfinite(y)]


y=y[x<1982]
x=x[x<1982]

N1=len(x)
k1=1+20.0/N1**2

mu1=mean(y)
S1=std(y,ddof=1)
sigma1=S1/sqrt(N1)

x=array(year)
y=array(density)
x=x[isfinite(y)]
y=y[isfinite(y)]


y=y[x>1982]
x=x[x>1982]

N2=len(x)
k2=1+20.0/N2**2

mu2=mean(y)
S2=std(y,ddof=1)
sigma2=S2/sqrt(N2)

x=array(year)
y=array(density)


# In[28]:

figure(figsize=(16,9))
plot(x,y,'-o')

mu,k,sigma=mu1,k1,sigma1
r=[1960,1981]
plot(r,[mu,mu],'g--')
plot(r,[mu+k*3*sigma,mu+k*3*sigma],'g:')
plot(r,[mu-k*3*sigma,mu-k*3*sigma],'g:')

mu,k,sigma=mu2,k2,sigma2
r=[1982,2014]
plot(r,[mu,mu],'r--')
plot(r,[mu+k*3*sigma,mu+k*3*sigma],'r:')
plot(r,[mu-k*3*sigma,mu-k*3*sigma],'r:')

s1=k1*sigma1
s2=k2*sigma2

mu=mu1-mu2
sigma=sqrt(s1**2+s2**2)

text(1990,8.0,r'$\mu_1-\mu_2 = %.4f \pm %.4f$' % (mu,sigma),fontsize=30)


xlabel('year')
ylabel('Density [g/cm$^3$]')
savefig('../../figs/density_fit3.pdf')


# In[29]:

volume_per_penny


# In[30]:

volume_per_penny


# In[31]:

x=array(year)
y=array(volume_per_penny)
x=x[isfinite(y)]
y=y[isfinite(y)]


y=y[x<1982]
x=x[x<1982]

N1=len(x)
k1=1+20.0/N1**2

mu1=mean(y)
S1=std(y,ddof=1)
sigma1=S1/sqrt(N1)

x=array(year)
y=array(volume_per_penny)
x=x[isfinite(y)]
y=y[isfinite(y)]


y=y[x>1982]
x=x[x>1982]

N2=len(x)
k2=1+20.0/N2**2

mu2=mean(y)
S2=std(y,ddof=1)
sigma2=S2/sqrt(N2)

x=array(year)
y=array(volume_per_penny)


# In[32]:

volume_per_penny


# In[33]:

figure(figsize=(16,9))
plot(x,y,'-o')

mu,k,sigma=mu1,k1,sigma1
r=[1960,1981]
plot(r,[mu,mu],'g--')
plot(r,[mu+k*3*sigma,mu+k*3*sigma],'g:')
plot(r,[mu-k*3*sigma,mu-k*3*sigma],'g:')

mu,k,sigma=mu2,k2,sigma2
r=[1982,2014]
plot(r,[mu,mu],'r--')
plot(r,[mu+k*3*sigma,mu+k*3*sigma],'r:')
plot(r,[mu-k*3*sigma,mu-k*3*sigma],'r:')

s1=k1*sigma1
s2=k2*sigma2

mu=mu1-mu2
sigma=sqrt(s1**2+s2**2)

text(1990,0.47,r'$\mu_1-\mu_2 = %.4f \pm %.4f$' % (mu,sigma),fontsize=30)

xlabel('year')
ylabel('Volume per Penny [cm$^3$]')
savefig('../../figs/volume_fit3.pdf')


# In[34]:

mu1-mu2


# In[35]:

s1=k1*3*sigma1
s2=k2*3*sigma2

mu=mu1-mu2
sigma=sqrt(s1**2+s2**2)

print mu,sigma


# In[35]:



