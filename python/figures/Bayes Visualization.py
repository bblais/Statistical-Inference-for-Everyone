
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
from bigfonts import *
import numpy as np
import pylab as pl


# In[3]:

np.random.seed(101)
p_disease=0.005
p_pos_given_disease=0.99
p_pos_given_no_disease=1-p_pos_given_disease  # doesn't have to be this
val_disease=100

a=255*np.ones((40,75))
r=np.random.rand(*a.shape)
r2=np.random.rand(*a.shape)


# In[4]:

a[r<p_disease]=val_disease


# In[5]:

figure(figsize=(21,11))
#pl.imshow(a,interpolation='nearest',cmap=pl.cm.gray,vmin=0,vmax=255)

#plot(10,10,'r+')
legend_count=0
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i,j]==val_disease:
            if legend_count==0:
                plot(j,i,'o',markersize=15,markerfacecolor='lightgray',label='Has the Disease')
                legend_count+=1
            else:
                plot(j,i,'o',markerfacecolor='lightgray',markersize=15)
            
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i,j]==val_disease:
            if r2[i,j]<p_pos_given_disease:
                if legend_count==1:
                    h=plot(j,i,'k.',markersize=15,label='Test Positive')
                    legend_count+=1
                else:
                    plot(j,i,'k.',markersize=15)
                
        else:
            if r2[i,j]<p_pos_given_no_disease:
                plot(j,i,'k.',markersize=15)
            
ax=gca()
ax.set_xlim([-0.5,a.shape[1]-0.5])
ax.set_ylim([-0.5,a.shape[0]-0.5])
ax.set_yticks(arange(a.shape[0])+0.5)
ax.set_xticks(arange(a.shape[1])+0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
legend()
grid(True)

savefig('../../figs/disease_plot.pdf',bbox_inches='tight')
            


# In[5]:

get_ipython().magic(u'pinfo plot')


# In[6]:

any(a.ravel())


# In[7]:

get_ipython().magic(u'pinfo imshow')


# In[8]:

prod(a.shape)


# In[9]:

get_ipython().magic(u'pinfo Circle')


# In[1]:

get_ipython().magic(u'pinfo plot')


# In[ ]:



