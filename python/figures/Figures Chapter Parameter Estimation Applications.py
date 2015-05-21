
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from sie import *


# In[2]:

dist=normal(2,.2)


# In[3]:

figure(figsize=(7,7))
x=linspace(-2,3,1000)
plot(x,dist.pdf(x))
gca().set_xticks([-2,-1,0,1,2,3])
gca().set_xticklabels(['','',0,'',r'$\mu$'+'\n(unknown true value)',''])
v=dist.pdf(2+.2)
plot([2,2+.2],[v,v],'k:')
arrow(2-.5,v,.5,0,length_includes_head=True,width=.002,linewidth=3,color='k')
arrow(2+.2+.5,v,-.5,0,length_includes_head=True,width=.002,linewidth=3,color='k')
text(.5,v,r'$\sigma$'+'\n(known deviation)',verticalalignment='center',horizontalalignment='center')
savefig('../../figs/normal_mu_known_sigma.pdf')


# In[3]:



