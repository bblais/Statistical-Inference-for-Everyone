from __future__ import with_statement,print_function
import pylab as py
import numpy as np
import emcee
from scipy.stats import distributions as D

greek=['alpha','beta','gamma','delta','chi','tau','mu',
        'sigma','lambda','epsilon','zeta','xi','theta','rho','psi']

def histogram(y,bins=50,plot=True):
    N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    if dx==0.0:  #  all in 1 bin!
        val=bins[0]
        bins=np.linspace(val-abs(val),val+abs(val),50)
        N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    x=bins[0:-1]+(bins[1]-bins[0])/2.0
    
    y=N*1.0/np.sum(N)/dx
    
    if plot:
        py.plot(x,y,'o-')
        yl=py.gca().get_ylim()
        py.gca().set_ylim([0,yl[1]])
        xl=py.gca().get_xlim()
        if xl[0]<=0 and xl[0]>=0:    
            py.plot([0,0],[0,yl[1]],'k--')

    return x,y


def corner(samples,labels):
    N=len(labels)
    from matplotlib.colors import LogNorm
    
    py.figure(figsize=(12,12))
    
    axes={}
    for i,l1 in enumerate(labels):
        for j,l2 in enumerate(labels):
            if j>i:
                continue
                
            ax = py.subplot2grid((N,N),(i, j))
            axes[(i,j)]=ax
            
            idx_y=labels.index(l1)
            idx_x=labels.index(l2)
            x,y=samples[:,idx_x],samples[:,idx_y]
            
            if i==j:
                # plot distributions
                xx,yy=histogram(x,bins=200,plot=False)
                py.plot(xx,yy,'-o',markersize=3)
                py.gca().set_yticklabels([])
                
                if i==(N-1):
                    py.xlabel(l2)
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                else:
                    ax.set_xticklabels([])
                
            else:
                counts,ybins,xbins,image = py.hist2d(x,y,bins=100,norm=LogNorm())
                #py.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)
                
                if i==(N-1):
                    py.xlabel(l2)
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                else:
                    ax.set_xticklabels([])
                    
                if j==0:
                    py.ylabel(l1)
                    [l.set_rotation(45) for l in ax.get_yticklabels()]
                else:
                    ax.set_yticklabels([])
    
    # make all the x- and y-lims the same
    j=0
    lims=[0]*N
    for i in range(1,N):
        ax=axes[(i,0)]
        lims[i]=ax.get_ylim()

        if i==N-1:
            lims[0]=ax.get_xlim()
    
        
    for i,l1 in enumerate(labels):
        for j,l2 in enumerate(labels):
            if j>i:
                continue
                
            ax=axes[(i,j)]
            
            if j==i:
                ax.set_xlim(lims[i])
            else:
                ax.set_ylim(lims[i])
                ax.set_xlim(lims[j])



import time

def time2str(tm):
    
    frac=tm-int(tm)
    tm=int(tm)
    
    s=''
    sc=tm % 60
    tm=tm//60
    
    mn=tm % 60
    tm=tm//60
    
    hr=tm % 24
    tm=tm//24
    dy=tm

    if (dy>0):
        s=s+"%d d, " % dy

    if (hr>0):
        s=s+"%d h, " % hr

    if (mn>0):
        s=s+"%d m, " % mn


    s=s+"%.2f s" % (sc+frac)

    return s

def timeit(reset=False):
    global _timeit_data
    try:
        _timeit_data
    except NameError:
        _timeit_data=time.time()

    if reset:
        _timeit_data=time.time()

    else:
        return time2str(time.time()-_timeit_data)


# In[6]:

from scipy.special import gammaln,gamma

def tpdf(x,df,mu,sd):
    t=(x-mu)/float(sd)
    return gamma((df+1)/2.0)/sqrt(df*pi)/gamma(df/2.0)/sd*(1+t**2/df)**(-(df+1)/2.0)
    
def logtpdf(x,df,mu,sd):
    try:
        N=len(x)
    except TypeError:
        N=1
    
    t=(x-mu)/float(sd)
    return N*(gammaln((df+1)/2.0)-0.5*np.log(df*np.pi)-gammaln(df/2.0)-np.log(sd))+(-(df+1)/2.0)*np.sum(np.log(1+t**2/df))

def loguniformpdf(x,mn,mx):
    if mn < x < mx:
        return np.log(1.0/(mx-mn))
    return -np.inf

def logjeffreyspdf(x):
    if x>0.0:
        return -np.log(x)
    return -np.inf

def logcauchypdf(x,x0,scale):
    return -np.log(np.pi)-np.log(scale)-np.log(1 + ((x-x0)/scale)**2)

def loghalfnormalpdf(x,sig):
    # x>0: 2/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
    if x<=0:
        return -np.inf
        
    return np.log(2)-0.5*np.log(2*np.pi*sig**2)*N - np.sum(x**2/sig**2/2.0)

def lognormalpdf(x,mn,sig):
    # 1/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
        
    return -0.5*np.log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)

def logexponpdf2(x,scale):
    if x<=0:
        return -np.inf
    return -np.log(scale)-x/scale

class BESTModel_OneSample(object):
    
    def __init__(self,y1):
        self.data=np.array(y1,float)
        pooled=self.data
        self.S=np.std(pooled)
        self.M=np.mean(pooled)
        
        self.names=['mu1','sigma1','nu']
        self.params=[]
        self.params_dict={}
        
    def initial_value(self):
        return np.array([self.M,self.S,10])
    
    def prior(self,theta):
        mu1,sd1,nu=theta
        value=0.0
        value+=lognormalpdf(mu1,self.M,1000*self.S)
        
        mn=0.001*self.S
        mx=1000*self.S
        value+=loguniformpdf(sd1,mn,mx-mn)

        value+=logexponpdf2(nu-1,scale=29)
        return value
        
    def run_mcmc(self,iterations=1000,burn=0.1):
        # Set up the sampler.
        ndim, nwalkers = len(self), 100
        val=self.initial_value()
        pos=emcee.utils.sample_ball(val, .05*val+1e-4, size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self)
        
        timeit(reset=True)
        print("Running MCMC...")
        self.sampler.run_mcmc(pos, iterations)
        print("Done.")
        print(timeit())
        
        burnin = int(self.sampler.chain.shape[1]*burn)
        samples = self.sampler.chain[:, burnin:, :]
        self.mu1=samples[:,:,0]
        self.sigma1=samples[:,:,1]
        self.nu=samples[:,:,2]
        
        self.params=[self.mu1,self.sigma1,self.nu]
        self.params_dict['mu1']=self.mu1
        self.params_dict['sigma1']=self.sigma1
        self.params_dict['nu']=self.nu
        
        
        
    def __len__(self):
        return 3  # mu1,sd1,nu
        
    def likelihood(self,theta):
        mu1,sd1,nu=theta
        y1=self.data
        
        value=0.0
        value+=logtpdf(y1,nu,mu1,sd1)

        return value
    
    def plot_chains(self,S=None,*args,**kwargs):
        if S is None:
            for S in ['mu1','sigma1','nu']:
                py.figure()
                self.plot_chains(S)
            return

        mu1,sigma1,nu=self.params
        N=float(np.prod(mu1.shape))

        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='mu1':
                name=r"\mu_1"
            elif name=='sigma1':
                name=r"\sigma_1"
            elif name=='nu':
                name=r"\nu"
            else:
                name=r"%s" % name
        
            value=eval(S)
        
        py.plot(value, color="k",alpha=0.02,**kwargs)
        if "\\" in name:        
            py.ylabel("$"+name+"$")        
        else:
            py.ylabel(name)        
    
    def plot_distribution(self,S=None,p=95):
        if S is None:
            for S in ['mu1','sigma1','nu']:
                py.figure()
                self.plot_distribution(S)
            return
        
        pp=[(100-p)/2.0,50,100-(100-p)/2.0]
        
        mu1,sigma1,nu=self.params
        N=float(np.prod(mu1.shape))
        
        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='mu1':
                name=r"\hat{\mu}_1"
            elif name=='sigma1':
                name=r"\hat{\sigma}_1"
            elif name=='nu':
                name=r"\hat{\nu}"
            else:
                name=r"\hat{%s}" % name
        
            value=eval(S)
            
        result=histogram(value.ravel(),bins=200)
        v=np.percentile(value.ravel(), pp ,axis=0)
        if r"\hat" in name:
            py.title(r'$%s=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        else:
            py.title(r'%s$=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        
        
    def P(self,S):
        
        mu1,sigma1,nu=self.params
        N=float(np.prod(mu1.shape))
        result=eval('np.sum(%s)/N' % S)
        return result
            
    def posterior(self,theta):
        prior = self.prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.likelihood(theta)
        
    def __call__(self,theta):
        return self.posterior(theta)


class BESTModel(object):
    
    def __init__(self,y1,y2):
        self.data=np.array(y1,float),np.array(y2,float)
        pooled=np.concatenate((y1,y2))
        self.S=np.std(pooled)
        self.M=np.mean(pooled)
        
        self.names=['mu1','mu2','sigma1','sigma2','nu']
        self.params=[]
        self.params_dict={}
        
    def initial_value(self):
        return np.array([self.M,self.M,self.S,self.S,10])
    
    def prior(self,theta):
        mu1,mu2,sd1,sd2,nu=theta
        value=0.0
        value+=lognormalpdf(mu1,self.M,1000*self.S)
        value+=lognormalpdf(mu2,self.M,1000*self.S)
        
        mn=0.001*self.S
        mx=1000*self.S
        value+=loguniformpdf(sd1,mn,mx-mn)
        value+=loguniformpdf(sd2,mn,mx-mn)

        value+=logexponpdf2(nu-1,scale=29)
        return value
        
    def run_mcmc(self,iterations=1000,burn=0.1):
        # Set up the sampler.
        ndim, nwalkers = len(self), 100
        val=self.initial_value()
        pos=emcee.utils.sample_ball(val, .05*val+1e-4, size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self)
        
        timeit(reset=True)
        print("Running MCMC...")
        self.sampler.run_mcmc(pos, iterations)
        print("Done.")
        print(timeit())
        
        burnin = int(self.sampler.chain.shape[1]*burn)
        samples = self.sampler.chain[:, burnin:, :]
        self.mu1=samples[:,:,0]
        self.mu2=samples[:,:,1]
        self.sigma1=samples[:,:,2]
        self.sigma2=samples[:,:,3]
        self.nu=samples[:,:,4]
        
        self.params=[self.mu1,self.mu2,self.sigma1,self.sigma2,self.nu]
        self.params_dict['mu1']=self.mu1
        self.params_dict['mu2']=self.mu2
        self.params_dict['sigma1']=self.sigma1
        self.params_dict['sigma2']=self.sigma2
        self.params_dict['nu']=self.nu
        
        
        
    def __len__(self):
        return 5  # mu1,mu2,sd1,sd2,nu
        
    def likelihood(self,theta):
        mu1,mu2,sd1,sd2,nu=theta
        y1,y2=self.data
        
        value=0.0
        value+=logtpdf(y1,nu,mu1,sd1)
        value+=logtpdf(y2,nu,mu2,sd2)

        return value
    
    def plot_chains(self,S=None,*args,**kwargs):
        if S is None:
            for S in ['mu1','mu2','sigma1','sigma2','nu']:
                py.figure()
                self.plot_chains(S)
            return

        mu1,mu2,sigma1,sigma2,nu=self.params
        N=float(np.prod(mu1.shape))

        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='mu1':
                name=r"\mu_1"
            elif name=='mu2':
                name=r"\mu_2"
            elif name=='sigma1':
                name=r"\sigma_1"
            elif name=='sigma2':
                name=r"\sigma_2"
            elif name=='nu':
                name=r"\nu"
            else:
                name=r"%s" % name
        
            value=eval(S)
        
        py.plot(value, color="k",alpha=0.02,**kwargs)
        if "\\" in name:        
            py.ylabel("$"+name+"$")        
        else:
            py.ylabel(name)        
    
    def plot_distribution(self,S=None,p=95):
        if S is None:
            for S in ['mu1','mu2','sigma1','sigma2','nu']:
                py.figure()
                self.plot_distribution(S)
            return
        
        pp=[(100-p)/2.0,50,100-(100-p)/2.0]
        
        mu1,mu2,sigma1,sigma2,nu=self.params
        N=float(np.prod(mu1.shape))
        
        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='mu1':
                name=r"\hat{\mu}_1"
            elif name=='mu2':
                name=r"\hat{\mu}_2"
            elif name=='sigma1':
                name=r"\hat{\sigma}_1"
            elif name=='sigma2':
                name=r"\hat{\sigma}_2"
            elif name=='nu':
                name=r"\hat{\nu}"
            else:
                name=r"\hat{%s}" % name
        
            value=eval(S)
            
        result=histogram(value.ravel(),bins=200)
        v=np.percentile(value.ravel(), pp ,axis=0)
        if r"\hat" in name:
            py.title(r'$%s=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        else:
            py.title(r'%s$=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        
        
    def P(self,S):
        
        mu1,mu2,sigma1,sigma2,nu=self.params
        N=float(np.prod(mu1.shape))
        result=eval('np.sum(%s)/N' % S)
        return result
            
    def posterior(self,theta):
        prior = self.prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.likelihood(theta)
        
    def __call__(self,theta):
        return self.posterior(theta)




from scipy.special import gammaln,gamma
def logfact(N):
    return gammaln(N+1)

def lognchoosek(N,k):
    return gammaln(N+1)-gammaln(k+1)-gammaln((N-k)+1)

def loguniformpdf(x,mn,mx):
    if mn < x < mx:
        return np.log(1.0/(mx-mn))
    return -np.inf

def logjeffreyspdf(x):
    if x>0.0:
        return -np.log(x)
    return -np.inf

def lognormalpdf(x,mn,sig):
    # 1/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
        
    return -0.5*np.log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)
    
def logbernoullipdf(theta, h, N):
    if 0.0<=theta<=1.0:
        return lognchoosek(N,h)+np.log(theta)*h+np.log(1-theta)*(N-h)
    else:
        return -np.inf

def logbetapdf(theta, h, N):
    if theta<0:
        return -np.inf
    if theta>1:
        return -np.inf

    if theta==0.0:
        if h==0:
            return logfact(N+1)-logfact(h)-logfact(N-h)+np.log(1-theta)*(N-h)
        else:
            return -np.inf
    elif theta==1.0:
        if (N-h)==0:
            return logfact(N+1)-logfact(h)-logfact(N-h)+np.log(theta)*h
        else:
            return -np.inf
    else:
        return logfact(N+1)-logfact(h)-logfact(N-h)+np.log(theta)*h+np.log(1-theta)*(N-h)

def logexponpdf(x,_lambda):
    # p(x)=l exp(l x)
    return _lambda*x + np.log(_lambda)

import scipy.optimize as op

class Normal(object):
    def __init__(self,mean=0,std=1):
        self.mean=mean
        self.std=std
        self.default=mean
        self.D=D.norm(mean,std)
        
    def rand(self,*args):
        return np.random.randn(*args)*self.std+self.mean
    
    def __call__(self,x):
        return lognormalpdf(x,self.mean,self.std)
class Exponential(object):
    def __init__(self,_lambda=1):
        self._lambda=_lambda
        self.D=D.expon(_lambda)

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logexponpdf(x,self._lambda)

class HalfNormal(object):
    def __init__(self,sigma=1):
        self.sigma=sigma
        self.D=D.halfnorm(sigma)

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return loghalfnormalpdf(x,self.sigma)

class Uniform(object):
    def __init__(self,min=0,max=1):
        self.min=min
        self.max=max
        self.default=(min+max)/2.0
        self.D=D.uniform(min,max-min)

    def rand(self,*args):
        return np.random.rand(*args)*(self.max-self.min)+self.min
        
    def __call__(self,x):
        return loguniformpdf(x,self.min,self.max)

class Jeffries(object):
    def __init__(self):
        self.default=1.0
        self.D=None # improper

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logjeffreyspdf(x)

class Cauchy(object):
    def __init__(self,x0=0,scale=1):
        self.x0=x0
        self.scale=scale
        self.default=x0
        self.D=D.cauchy(loc=x0,scale=scale) 

    def rand(self,*args):
        return np.random.rand(*args)*2-1
        
    def __call__(self,x):
        return logcauchypdf(x,self.x0,self.scale)


class Beta(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N
        a=h+1
        b=(N-h)+1
        self.D=D.beta(a,b)

    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbetapdf(x,self.h,self.N)

class Bernoulli(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N
        self.D=D.bernoulli(self.default)

    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbernoullipdf(x,self.h,self.N)
     

def lnprior_function(model):
    def _lnprior(x):
        return model.lnprior(x)

    return _lnprior

class MCMCModel_Meta(object):

    def __init__(self,**kwargs):
        self.params=kwargs
        
        self.keys=[]
        for key in self.params:
            self.keys.append(key)


        self.index={}
        for i,key in enumerate(self.keys):
            self.index[key]=i


        self.nwalkers=100
        self.burn_percentage=0.25
        self.initial_value=None
        self.samples=None
        self.last_pos=None

    def lnprior(self,theta):
        pass

    def lnlike(self,theta):
        pass

    def lnprob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)        

    def __call__(self,theta):
        return self.lnprob(theta)

    def set_initial_values(self,method='prior',*args,**kwargs):
        if method=='prior':
            ndim=len(self.params)
            try:
                N=args[0]
            except IndexError:
                N=300

            pos=np.zeros((self.nwalkers,ndim))
            for i,key in enumerate(self.keys):
                pos[:,i]=self.params[key].rand(self.nwalkers)

            
            self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, 
                    lnprior_function(self))

            timeit(reset=True)
            print("Sampling Prior...")
            self.sampler.run_mcmc(pos, N,**kwargs)
            print("Done.")
            print( timeit())

            # assign the median back into the simulation values
            self.burn()
            self.median_values=np.percentile(self.samples,50,axis=0)

            self.last_pos=self.sampler.chain[:,-1,:]
        elif method=='samples':
            lower,upper=np.percentile(self.samples, [16,84],axis=0)            
            subsamples=self.samples[((self.samples>=lower) & (self.samples<=upper)).all(axis=1),:]
            idx=np.random.randint(subsamples.shape[0],size=self.last_pos.shape[0])
            self.last_pos=subsamples[idx,:]            


        elif method=='maximum likelihood':
            self.set_initial_values()
            chi2 = lambda *args: -2 * self.lnlike_lownoise(*args)
            result = op.minimize(chi2, self.initial_value)
            vals=result['x']
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        elif method=='median':            
            vals=self.median_values
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        else:
            raise ValueError("Unknown method: %s" % method)

    def burn(self,burn_percentage=None):
        if not burn_percentage is None:
            self.burn_percentage=burn_percentage
            
        if self.burn_percentage>1:
            self.burn_percentage/=100.0

        burnin = int(self.sampler.chain.shape[1]*self.burn_percentage)  # burn 25 percent
        ndim=len(self.params)
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    def run_mcmc(self,N,**kwargs):
        ndim=len(self.params)
        
        if self.last_pos is None:
            self.set_initial_values()
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self,)
        
        timeit(reset=True)
        print("Running MCMC...")
        self.sampler.run_mcmc(self.last_pos, N,**kwargs)
        print("Done.")
        print (timeit())

        # assign the median back into the simulation values
        self.burn()
        self.median_values=np.percentile(self.samples,50,axis=0)
        theta=self.median_values

        self.last_pos=self.sampler.chain[:,-1,:]


    def plot_chains(self,*args,**kwargs):
        py.clf()
        
        if not args:
            args=self.keys
        
        
        fig, axes = py.subplots(len(args), 1, sharex=True, figsize=(8, 5*len(args)))
        try:  # is it iterable?
            axes[0]
        except TypeError:
            axes=[axes]



        labels=[]
        for ax,key in zip(axes,args):
            i=self.index[key]
            sample=self.sampler.chain[:, :, i].T

            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                try:
                    label = r'$%s$ - %s' % (namestr,self.column_names[i])
                except (AttributeError,IndexError):
                    label='$%s$' % namestr
                # label='$%s$' % namestr

            labels.append(label)
            ax.plot(sample, color="k", alpha=0.2,**kwargs)
            ax.set_ylabel(label)

    def triangle_plot(self,*args,**kwargs):
        
        if not args:
            args=self.keys
            
        assert len(args)>1
        
        labels=[]
        idx=[]
        for key in args:
            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key


                try:
                    label = r'$%s$ - %s' % (namestr,self.column_names[i])
                except (AttributeError,IndexError):
                    label='$%s$' % namestr
                    

            labels.append(label)
            idx.append(self.index[key])
        
        fig = corner(self.samples[:,idx], labels=labels,**kwargs)

            
    def plot_distributions(self,*args,**kwargs):
        if not args:
            args=self.keys
        
        for key in args:
            if key.startswith('_sigma'):
                label=r'\sigma'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='%s' % namestr

            i=self.index[key]
            
            py.figure(figsize=(12,4))
            result=histogram(self.samples[:,i],bins=200)
            xlim=py.gca().get_xlim()
            x=py.linspace(xlim[0],xlim[1],500)
            y=D.norm.pdf(x,np.median(self.samples[:,i]),np.std(self.samples[:,i]))
            py.plot(x,y,'-')

            v=np.percentile(self.samples[:,i], [2.5, 50, 97.5],axis=0)

            if v[1]<.005 or (v[2]-v[1])<0.005 or (v[1]-v[0])<0.005:
                py.title(r'$\hat{%s}^{97.5}_{2.5}=%.3g^{+%.3g}_{-%.3g}$' % (label,v[1],(v[2]-v[1]),(v[1]-v[0])))
            else:
                py.title(r'$\hat{%s}^{97.5}_{2.5}=%.3f^{+%.3f}_{-%.3f}$' % (label,v[1],(v[2]-v[1]),(v[1]-v[0])))
            py.ylabel(r'$p(%s|{\rm data})$' % label)
            try:
                py.xlabel(r'$%s$ - %s' % (label,self.column_names[i]))
            except (AttributeError,IndexError):
                py.xlabel(r'$%s$' % label)

    def plot_distributions_K(self,*args,**kwargs):
        if not args:
            args=self.keys
        
        for key in args:
            if key.startswith('_sigma'):
                label=r'\sigma'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='%s' % namestr

            i=self.index[key]
            
            py.figure(figsize=(12,4))

            x,y=histogram(self.samples[:,i],bins=200,plot=False)
            py.plot(x,y,'.-')
            py.fill_between(x,y,facecolor='blue', alpha=0.2)

            HDI=np.percentile(self.samples[:,i], [2.5, 50, 97.5],axis=0)
            yl=py.gca().get_ylim()
            py.text((HDI[0]+HDI[2])/2, 0.15*yl[1],'95% HDI', ha='center', va='center',fontsize=12)
            py.plot(HDI,[yl[1]*.1,yl[1]*.1,yl[1]*.1],'k.-',linewidth=1)
            for v in HDI:
                if v<0.005:
                    py.text(v, 0.05*yl[1],'%.3g' % v, ha='center', va='center', 
                         fontsize=12)
                else:
                    py.text(v, 0.05*yl[1],'%.3f' % v, ha='center', va='center', 
                         fontsize=12)

            py.ylabel(r'$p(%s|{\rm data})$' % label)

            try:
                py.xlabel(r'$%s$ - %s' % (label,self.column_names[i]))
            except (AttributeError,IndexError):
                py.xlabel(r'$%s$' % label)

                
    def get_distribution(self,key,bins=200):
            
        i=self.index[key]
        x,y=histogram(self.samples[:,i],bins=bins,plot=False)
        
        return x,y
        
    def percentiles(self,p=[16, 50, 84]):
        result={}
        for i,key in enumerate(self.keys):
            result[key]=np.percentile(self.samples[:,i], p,axis=0)
            
        return result
        
    def best_estimates(self):
        self.median_values=np.percentile(self.samples,50,axis=0)
        theta=self.median_values
        
        return self.percentiles()
    
    def get_samples(self,*args):
        result=[]
        for arg in args:
            idx=self.keys.index(arg)
            result.append(self.samples[:,idx])
    
        return tuple(result)
    
    def eval(self,S):
        for i,key in enumerate(self.keys):
            exec('%s=self.samples[:,i]' % key)
        result=eval(S)
        return result
    
    def P(self,S):
        
        for i,key in enumerate(self.keys):
            exec('%s=self.samples[:,i]' % key)
            
        N=float(np.prod(self.samples[:,0].shape))
        result=eval('np.sum(%s)/N' % S)
        return result
 

class MCMCModel(MCMCModel_Meta):
    def __init__(self,data,P_data,**kwargs):

        self.data=data
        self.params=kwargs
        
        self.lnlike_function=P_data

        MCMCModel_Meta.__init__(self,**kwargs)

    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            params_dict[key]=theta[i]
                
        return self.lnlike_function(self.data,**params_dict)

class MCMCModel_Regression(MCMCModel_Meta):
    
    def __init__(self,x,y,function,**kwargs):
        self.x=x
        self.y=y
        self.function=function
        self.params=kwargs
        
        MCMCModel_Meta.__init__(self,**kwargs)

        self.params['_sigma']=Jeffries()
        self.keys.append('_sigma')        
        self.index['_sigma']=len(self.keys)-1


    # Define the probability function as likelihood * prior.
    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=theta[i]
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)
    
    def lnlike_lownoise(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=1.0
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)


    def predict(self,x,theta=None):
        
        if theta is None:
            self.percentiles()
            theta=self.median_values
            
        args={}
        for v,key in zip(theta,self.keys):
            if key=='_sigma':
                continue
            args[key]=v

        y_predict=np.array([self.function(_,**args) for _ in x])        
        return y_predict
    
    def plot_predictions(self,x,N=1000,color='k'):
        samples=self.samples[-N:,:]
        
        for value in samples:
            args={}
            for v,key in zip(value,self.keys):
                if key=='_sigma':
                    continue
                args[key]=v

            y_predict=np.array([self.function(_,**args) for _ in x])        
            py.plot(x,y_predict,color=color,alpha=0.02)

import patsy

class MCMCModel_MultiLinear(MCMCModel_Meta):
    
    def __init__(self,data,eqn,**kwargs):
        
        self.dmatrices=patsy.dmatrices(eqn, data)
        self.eqn=eqn
        
        self.y=np.array(self.dmatrices[0])
        self.X=np.array(self.dmatrices[1])
        self.column_names=self.dmatrices[1].design_info.column_names
        
            
        MCMCModel_Meta.__init__(self,**kwargs)

        self.index={}
        self.keys=[]
        self.params={}
        count=0
        for i,paramname in enumerate(
                    ['beta_%d' % _ for _ in range(len(self.column_names))]):
            col=self.column_names[i]
            if paramname in kwargs:
                self.params[paramname]=kwargs[paramname]
            elif col in kwargs:
                self.params[paramname]=kwargs[col]
            else:
                self.params[paramname]=Normal(0,10)
            self.index[paramname]=i
            self.keys.append(paramname)
            count+=1
        
        if 'sigma' in kwargs:
            self.params['_sigma']=kwargs['sigma']
        else:
            self.params['_sigma']=Jeffries()

        self.keys.append('_sigma')        
        self.index['_sigma']=len(self.keys)-1
            

    # Define the probability function as likelihood * prior.
    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        # params_dict={}
        # for i,key in enumerate(self.keys):
        #     if key=='_sigma':
        #         sigma=theta[i]
        #     else:
        #         params_dict[key]=theta[i]
                
        y_fit=self.X*theta[:-1]
        
        return lognormalpdf(self.y,y_fit,sigma)
    

    def summary(self):
        from pandas import DataFrame
        from IPython.display import display

        sdata={}
        names=[]
        sdata['median']=[]
        sdata['2.5%']=[]
        sdata['97.5%']=[]
        p=self.percentiles([2.5,50,97.5])
        for key,col in zip(self.keys,self.column_names+['']):
            name=key+" - "+col
            names.append(name)
            sdata['median'].append(p[key][1])
            sdata['2.5%'].append(p[key][0])
            sdata['97.5%'].append(p[key][2])

        sdf=DataFrame(sdata,index=names,columns=['2.5%','median','97.5%'])
        display(sdf)
        

    def predict(self,theta=None,**kwargs):
        
        if theta is None:
            self.percentiles()
            theta=self.median_values
            
        data=pandas.DataFrame(kwargs)
        dmatrices=patsy.dmatrices(self.eqn, data)
        
        y=np.array(dmatrices[0])
        X=np.array(dmatrices[1])

        y_fit=X*theta[:-1]

        return y_fit
    
    def plot_predictions(self,x,N=1000,color='k'):
        samples=self.samples[-N:,:]

        for theta in samples:
            y_fit=self.X*theta[:-1]
            py.plot(x,y_predict,color=color,alpha=0.02)
