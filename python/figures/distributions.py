from plotty import *

from utils import distplot,distplot2
from scipy.stats import distributions as D

def uniform(min=0,max=1):
    return D.uniform(min,max-min)

def gaussian(mu=0,sd=1):
    return D.norm(mu,sd)




for fig in [1,2,3,4]:

    if fig==1:
        dist=uniform(min=0,max=1)
        distplot(dist,xlim=[-.2,1.2],fignum=fig,
            quartiles=array([5,25,50,75,95])/100.0,
            label='x')
        title('Min=0, Max=1')
    elif fig==2:
        dist=uniform(min=32,max=42)        
        distplot(dist,xlim=[30,44],fignum=fig,
            quartiles=array([5,25,50,75,95])/100.0,
            label='x')
        title('Min=32, Max=42')
    elif fig==3:
        dist=gaussian(0,1)
        distplot(dist,fignum=fig,
            label='x')
        title(r'$\mu=0, \sigma=1$',fontsize=30)
    elif fig==4:
        dist1=gaussian(0,1)
        dist2=gaussian(0,2)
        distplot2([dist1,dist2],fignum=fig,
            xlim=[-6,6],
            quartiles=array([5,23,50,67,95])/100.0,
            label='x')
        title(r'$\mu=0, \sigma=1,\sigma=2$',fontsize=30)
    

    draw()
    
    savefig('distributions_%d.pdf' % fig)



