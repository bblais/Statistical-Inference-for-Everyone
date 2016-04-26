
# coding: utf-8

# # Distributions

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

from sie import *


# In[2]:

dist=uniform(min=0,max=1)
figure(figsize=(10,8))
distplot(dist,xlim=[-.2,1.2],
    quartiles=array([5,25,50,75,95])/100.0,
    label='x',
    notebook=True)
title('Min=0, Max=1')
savefig('../../figs/distributions_1.pdf')


# In[3]:

dist=uniform(min=0,max=4)
figure(figsize=(10,8))
distplot(dist,xlim=[-.2,4.2],
    quartiles=array([5,25,50,75,95])/100.0,
    label='time',
    fill_between_values=[2,2+1.0/3],
    notebook=True)
title('Min=0, Max=4')
text(2.5,.1,'20 minutes',verticalalignment='center')
arrow( 2.4, 0.1, 2.2-2.4, 0.0, fc="k", ec="k",
head_width=0.008, head_length=0.08 )

savefig('../../figs/distributions_plumber.pdf')


# In[2]:

dist=beta(h=3,N=12)
figure(figsize=(10,8))
distplot(dist,xlim=[-.1,1.1],
    quartiles=[],
    label=r'$\theta$',
    fill_between_quartiles=[0,1],
    notebook=True)
grid(True)
gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
plot([.25],[dist.pdf(.25)],'o',markersize=10)

ax=gca()
vx=0.25
vy=dist.pdf(.25)
ax.annotate('maximum probability', xy=(vx, vy),  xycoords='data',
            xytext=(-50, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            color=[0,.4,0],  # green
            )


title('3 heads and 9 tails')
savefig('../../figs/beta_dist.pdf',transparent=False)


# In[2]:

dist=beta(h=3,N=12)
figure(figsize=(10,8))
distplot(dist,xlim=[-.1,1.1],
    quartiles=[],
    label=r'$\theta$',
    fill_between_values=[0,0.5],
    notebook=True)
grid(True)
gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

ax=gca()


area=dist.cdf(0.5)
ax.annotate('area=%.3f' % area, xy=(0.3, 1.5),  xycoords='data',
            xytext=(60, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            color=[0,.4,0],  # green
            )

title('3 heads and 9 tails')
savefig('../../figs/beta_dist2.pdf')


# In[2]:

dist=beta(h=3,N=12)
figure(figsize=(10,8))
distplot(dist,xlim=[-.1,1.1],
    quartiles=[],
    label=r'$\theta$',
    fill_between_quartiles=[0,0.5],
    values=[dist.ppf(0.5)],
    notebook=True)
grid(True)
gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

ax=gca()


area=0.5
ax.annotate('area=%.1f' % area, xy=(0.25, 2.8),  xycoords='data',
            xytext=(60, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            color=[0,.4,0],  # green
            )

title('3 heads and 9 tails')
savefig('../../figs/beta_dist3.pdf')


# In[7]:

dist.ppf(0.5)


# In[8]:

dist=beta(h=3,N=12)
figure(figsize=(10,8))
distplot(dist,xlim=[-.1,1.1],
    quartiles=array([1,5,25,50,75,95,99])/100.0,
    label=r'$\theta$',
    notebook=True)
grid(True)
gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

ax=gca()

title('3 heads and 9 tails')
savefig('../../figs/beta_dist4.pdf')

table="<pre>\n"
table+=r"\begin{tabular}{cc}"+"\n"
table+=r"\multicolumn{2}{c}{\textit{\textbf{Beta({\rm heads}=3,{\rm tails}=9)}}} \\" + "\n"
table+=r"{\bf Value} & {\bf Area} \\"+"\n"
for area in [0.01,0.05,0.1,0.25,0.50,0.75,0.90,0.95,0.99]:
    value=dist.ppf(area)
    table+=r"%.2f & %.2f \\" % (value,area)
    table+="\n"
table+=r"\end{tabular}"+"\n"
table+="</pre>\n"

IPython.display.HTML(table)



# In[9]:

h=10
N=30
dist=beta(h=h,N=N)
figure(figsize=(10,8))
distplot(dist,xlim=[0.0,0.8],
    quartiles=array([1,5,25,50,75,95,99])/100.0,
    label=r'$\theta$',
    notebook=True)
grid(True)
gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8])

ax=gca()

title('%d heads and %d tails' % (h,N-h))
savefig('../../figs/beta_dist5.pdf')

table="<pre>\n"
table+=r"\begin{tabular}{cc}"+"\n"
table+=r"\multicolumn{2}{c}{\textit{\textbf{Beta({\rm heads}=%d,{\rm tails}=%d)}}} \\" % (h,N-h)
table+="\n"
table+=r"{\bf Value} & {\bf Area} \\"+"\n"
for area in [0.01,0.05,0.1,0.25,0.50,0.75,0.90,0.95,0.99]:
    value=dist.ppf(area)
    table+=r"%.2f & %.2f \\" % (value,area)
    table+="\n"
table+=r"\end{tabular}"+"\n"
table+="</pre>\n"

IPython.display.HTML(table)


# ## Normal Distribution Exploration

# In[10]:

distplot(gaussian(0,1),show_quartiles=False)
plot([-4,4],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x) = {\rm Normal}(0,1)$')
savefig('../../figs/gaussian1.pdf')


# In[11]:

for mu in [0,3,-2]:
    distplot(gaussian(mu,1),show_quartiles=False,notebook=True,xlim=[-6,6])
    
    text(mu,0.41,r'$\mu=%d$' % mu,horizontalalignment='center',size=25)
plot([-6,6],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x) = {\rm Normal}(\mu,1)$')
savefig('../../figs/gaussian_mu.pdf')


# In[12]:

for y,sigma in zip([0.11,0.21,0.41],[4,2,1]):
    distplot(gaussian(0,sigma),show_quartiles=False,notebook=True,xlim=[-8,8])
    
    text(0,y,r'$\sigma=%d$' % sigma,horizontalalignment='center',size=25)
plot([-6,6],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x) = {\rm Normal}(0,\sigma)$')
savefig('../../figs/gaussian_sigma.pdf')


# In[13]:

mu_x=8; sigma_x=2; 
distplot(gaussian(mu_x,sigma_x),show_quartiles=False,notebook=True,xlim=[-10,40])
mu_y=20; sigma_y=7; 
distplot(gaussian(mu_y,sigma_y),show_quartiles=False,notebook=True,xlim=[-10,40])
    
mu_z=mu_y-mu_x
sigma_z=sqrt(sigma_x**2+sigma_y**2)
distplot(gaussian(mu_z,sigma_z),show_quartiles=False,notebook=True,xlim=[-10,40])

text(mu_x,.21,r'$p(x)={\rm Normal}(%d,%d)$' % (mu_x,sigma_x),horizontalalignment='center',size=25,color='blue')
text(mu_y,.07,r'$p(y)={\rm Normal}(%d,%d)$' % (mu_y,sigma_y),horizontalalignment='center',size=25,color='green')
text(mu_z,-.02,r'$p(z)=p(y-x)={\rm Normal}(%.1f,%.1f)$' % (mu_z,sigma_z),horizontalalignment='center',size=25,color='red')

gca().set_ylim([-.03,.23])
plot([0,30],[0,0],'k--')
#xlabel('')
#ylabel(r'$p(x) = {\rm Normal}(0,\sigma)$')
savefig('../../figs/gaussian_sigma_y_minus_x.pdf')


# In[13]:




# In[13]:




# In[14]:

distplot(gaussian(0,1),show_quartiles=True,
    quartiles=array([50-99.73/2,50-95.4/2,50-68.2/2,50,68.2/2+50,95.4/2+50,99.73/2+50])/100.0,
    fill_between_values=[-1,1],
    quartile_format='{:.1%}',
)


plot([-4,4],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x) = {\rm Normal}(0,1)$')
f=gcf()
ax=gca()

if True:
    c=ax.get_children()
    c=[_ for _ in c if type(_)==matplotlib.text.Text]
    for t in c:
        if '%' in t.get_text():
            pos=t.get_position()
            pos=[pos[0]+0.1,pos[1]+0.015]
            t.set_position(pos)
            
            if t.get_text()=='50.0%':
                pos=t.get_position()
                pos=[pos[0],pos[1]-.04]
                t.set_position(pos)
                
        
area=0.68
ax.annotate('area=%.2f' % area, xy=(0.25, .2),  xycoords='data',
            xytext=(90, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            color=[0,.4,0],  # green
            )
        
savefig('../../figs/gaussian_standard.pdf')


# In[15]:

t=c[0]
t.get_position()


# ## Beta Distribution

# In[16]:

distplot(beta(3,12),show_quartiles=False,xlim=[0,1])
plot([-4,4],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x) = {\rm Beta}(h=3,N=12)$')
savefig('../../figs/beta1.pdf')


# In[17]:

h=6
N=12.0
mu=h/N
sig=sqrt( mu*(1-mu)/N)
distplot(beta(h,N),show_quartiles=False,xlim=[0,1],notebook=True)
distplot(normal(mu,sig),show_quartiles=False,xlim=[0,1],notebook=True)
plot([-4,4],[0,0],'k--')
xlabel('$x$')
ylabel(r'$p(x)$')
legend(['Beta','Normal'])
savefig('../../figs/beta2.pdf')


# In[18]:

import sie


# In[19]:

sie.__file__


# In[20]:

pwd


# In[21]:

d=gaussian(0,1)


# In[22]:

d.cdf(-1)-d.cdf(1)


# In[23]:

d.cdf(-3)-d.cdf(3)


# In[24]:

d.cdf(-2)-d.cdf(2)


# ## Visualization

# In[25]:

data=load_data('../../python/rdatasets/csv/MASS/survey.csv')
male_data=data[data['Sex']=='Male']
female_data=data[data['Sex']=='Female']
male_height=male_data['Height'].dropna()
female_height=female_data['Height'].dropna()


# In[26]:

bins=linspace(140,210,20)
hist(male_height.dropna(),bins,alpha=0.8)
xlabel('Height [cm]')
ylabel('Number of People')
savefig('../../figs/histmale.pdf')


# In[27]:

mean(male_height)


# In[28]:

len(male_height)


# In[29]:

min(male_height)


# In[30]:

max(male_height)


# In[31]:

subdata=male_data[['Height','Wr.Hnd']].dropna()


# In[32]:

height=subdata['Height']
wr_hand=subdata['Wr.Hnd']


# In[33]:

figure()
bins=linspace(140,210,20)
hist(height,bins,alpha=0.8)
xlabel('Height [cm]')
ylabel('Number of People')
savefig('../../figs/height_hist.pdf')

figure()
hist(wr_hand,bins=15,alpha=0.8)
xlabel('Writing Hand Span [cm]')
ylabel('Number of People')
savefig('../../figs/wrhand_hist.pdf')

figure()
plot(height,wr_hand,'o')
ylabel('Writing Hand Span [cm]')
xlabel('Height [cm]')
gca().set_ylim([15,24])
gca().set_xlim([150,210])

result=pandas.ols(y=wr_hand,x=height)
m,b=result.beta['x'],result.beta['intercept']
xx=linspace(155,205,20)
yy=m*xx+b
plot(xx,yy,'b--')


savefig('../../figs/height_wrhand_scatter.pdf')


# In[34]:

len(height)


# In[35]:

result=pandas.ols(y=wr_hand,x=height)


# In[36]:

m,b=result.beta['x'],result.beta['intercept']


# In[37]:

bins=linspace(140,210,4)
hist(male_height.dropna(),bins,alpha=0.8)
xlabel('Height [cm]')
ylabel('Number of People')
savefig('../../figs/histmale_toofewbins.pdf')


# In[38]:

bins=linspace(140,210,105)
hist(male_height.dropna(),bins,alpha=0.8)
xlabel('Height [cm]')
ylabel('Number of People')
savefig('../../figs/histmale_toomanybins.pdf')


# ## Normal vs Beta

# In[39]:

for h,N in [ (3,12),(30,120),(5,10),(50,100) ]:

    figure(figsize=(10,8))
    dist=beta(h=h,N=N)
    distplot(dist,xlim=[-.1,1.1],
        quartiles=[],
        label=r'$\theta$',
        notebook=True)
    f=float(h)/N
    mu=f
    sd=sqrt(f*(1-f)/N)
    
    dist=normal(mu,sd)
    distplot(dist,xlim=[-.1,1.1],
        quartiles=[],
        label=r'$\theta$',
        notebook=True)
    
    grid(True)
    gca().set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.7, 0.6), xycoords='axes fraction')
    
    legend(['Beta','Normal'])
    title('%d heads and %d tails' % (h,N-h))
    savefig('../../figs/beta_normal_%d_%d.pdf' % (h,N) )


# In[40]:

zip( [3,12],[30,120],[1,10])


# ## Poisson

# In[41]:

_lambda=4
dist=D.poisson(_lambda)
figure(figsize=(10,8))

k=arange(0,13)
y=dist.pmf(k)
gca().set_xlim([-1,12])
plot(k,y,'--o',linewidth=3,markersize=15)
xlabel('$k$')
ylabel(r'${\rm Poisson}(\lambda=%d)$' % _lambda)
savefig('../../figs/distributions_poisson.pdf')


# In[42]:

_lambda=4
dist=D.poisson(_lambda)
figure(figsize=(10,8))

k=arange(0,13)
y=dist.pmf(k)
gca().set_xlim([-1,12])
plot(k,y,'b--',linewidth=3,markersize=15,label='Poisson')
plot(k,y,'bo',linewidth=3,markersize=15)

dist=normal(mu=_lambda,sd=sqrt(_lambda))

xx=linspace(-1,13,28*3+1)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'g-',linewidth=3,label='Normal')

xx=k
mu=_lambda
sd=sqrt(_lambda)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'go',markersize=10)

xlabel('$k$')
ylabel(r'${\rm Poisson}(\lambda=%d)$' % _lambda)
legend(loc='best')

annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.75, 0.6), xycoords='axes fraction')

savefig('../../figs/poisson_normal1.pdf')



# In[43]:

_lambda=20
max_k=40
dist=D.poisson(_lambda)
figure(figsize=(10,8))

k=arange(0,max_k+1)
y=dist.pmf(k)
gca().set_xlim([-1,max_k])
plot(k,y,'b--',linewidth=3,markersize=15,label='Poisson')
plot(k,y,'bo',linewidth=3,markersize=15)

dist=normal(mu=_lambda,sd=sqrt(_lambda))

xx=linspace(-1,max_k+1,(2*(max_k+1))*3+1)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'g-',linewidth=3,label='Normal')

xx=k
mu=_lambda
sd=sqrt(_lambda)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'go',markersize=10)

xlabel('$k$')
ylabel(r'${\rm Poisson}(\lambda=%d)$' % _lambda)
legend(loc='best')

annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.75, 0.6), xycoords='axes fraction')

savefig('../../figs/poisson_normal2.pdf')


# In[44]:

_lambda=50
max_k=100
dist=D.poisson(_lambda)
figure(figsize=(10,8))

k=arange(0,max_k+1)
y=dist.pmf(k)
gca().set_xlim([-1,max_k])
plot(k,y,'b--',linewidth=3,markersize=15,label='Poisson')
plot(k,y,'bo',linewidth=3,markersize=15)

dist=normal(mu=_lambda,sd=sqrt(_lambda))

xx=linspace(-1,max_k+1,(2*(max_k+1))*3+1)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'g-',linewidth=3,label='Normal')

xx=k
mu=_lambda
sd=sqrt(_lambda)
yy=normal(mu=_lambda,sd=sqrt(_lambda)).pdf(xx)
plot(xx,yy,'go',markersize=10)

xlabel('$k$')
ylabel(r'${\rm Poisson}(\lambda=%d)$' % _lambda)
legend(loc='best')

annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.75, 0.6), xycoords='axes fraction')

savefig('../../figs/poisson_normal3.pdf')


# ## Binomial

# In[45]:

N=10
p=0.25

dist=D.binom(N,p)
figure(figsize=(10,8))

k=arange(0,N+1)
y=dist.pmf(k)
gca().set_xlim([-1,12])
plot(k,y,'b--',linewidth=3,markersize=15,label='Binomial')
plot(k,y,'bo',linewidth=3,markersize=15)

mu=p*N
sd=sqrt(N*p*(1-p))
dist=normal(mu=mu,sd=sd)

xx=linspace(-1,N+1,(N+1)*2*3+1)
yy=normal(mu=mu,sd=sd).pdf(xx)
plot(xx,yy,'g-',linewidth=3,label='Normal')

xx=k
yy=normal(mu=mu,sd=sd).pdf(xx)
plot(xx,yy,'go',markersize=10)

xlabel('$k$')
ylabel(r'${\rm Binomial}(N=%d,p=%.2f)$' % (N,p))
legend(loc='best')

annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.75, 0.6), xycoords='axes fraction')

savefig('../../figs/binomial_normal1.pdf')


# In[46]:

N=100
p=0.25

dist=D.binom(N,p)
figure(figsize=(10,8))

k=arange(0,N+1)
y=dist.pmf(k)
gca().set_xlim([-1,N])
plot(k,y,'b--',linewidth=3,markersize=15,label='Binomial')
plot(k,y,'bo',linewidth=3,markersize=15)

mu=p*N
sd=sqrt(N*p*(1-p))
dist=normal(mu=mu,sd=sd)

xx=linspace(-1,N+1,(N+1)*2*3+1)
yy=normal(mu=mu,sd=sd).pdf(xx)
plot(xx,yy,'g-',linewidth=3,label='Normal')

xx=k
yy=normal(mu=mu,sd=sd).pdf(xx)
plot(xx,yy,'go',markersize=10)

xlabel('$k$')
ylabel(r'${\rm Binomial}(N=%d,p=%.2f)$' % (N,p))
legend(loc='best')

annotate(r'$\mu=%.2f$' % mu +'\n'+'$\sigma=%.2f$' % sd, fontsize=30,
                xy=(0.75, 0.6), xycoords='axes fraction')

savefig('../../figs/binomial_normal2.pdf')


# In[46]:



