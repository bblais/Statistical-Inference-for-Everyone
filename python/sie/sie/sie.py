from __future__ import with_statement,print_function
import os
import pandas
from pylab import *
from numpy import *

import numpy as np
import IPython.display
from Struct import Struct

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['figure.figsize'] = (12,8)
mpl.rcParams['axes.grid']=True
mpl.rcParams['figure.dpi']=300

size=20
family='sans-serif'
    
from matplotlib import rc
from scipy import stats
from scipy.stats import distributions as D


def uniform(min=0,max=1):
    return D.uniform(min,max-min)

def gaussian(mu=0,sd=1):
    return D.norm(mu,sd)


def normal(mu=0,sd=1):
    return D.norm(mu,sd)

def tdist(df,mu=0,sd=1):
    return D.t(df,mu,sd)

def coinflip(h,N):
    a=h+1
    b=(N-h)+1
    return D.beta(a,b)

def beta(h=5,N=10):
    a=h+1
    b=(N-h)+1
    return D.beta(a,b)

def sample_deviation(x):
    return std(x,ddof=1)
    
def sample_mean(x):
    return mean(x)


def credible_interval(dist,percentage=95):
    try:
        c0=percentage[0]/100.0
        c1=percentage[1]/100.0
    except TypeError:
        c0=(1-percentage/100.0)/2
        c1=percentage/100.0+(1-percentage/100.0)/2
    except IndexError:
        c0=(1-percentage[0]/100.0)/2
        c1=percentage[0]+(1-percentage[0]/100.0)/2
        
    bottom=dist.ppf(c0)
    med=dist.ppf(0.5)
    top=dist.ppf(c1)

    return bottom,med,top


def credible_interval_plot(dist,percentage=95,xlim=None):

    CI=credible_interval(dist,percentage)

    if xlim is None:
        xlim=gca().get_xlim()

    plot(xlim,[CI[0],CI[0]],'g:')
    plot(xlim,[CI[1],CI[1]],'g--')
    plot(xlim,[CI[2],CI[2]],'g:')

def distplot_K(x,dist,ylim=None):
    from pylab import plot,fill_between,gca,text
    
    try:
        y=dist.pdf(x)
    except AttributeError:
        y=dist
        
    plot(x,y)
    fill_between(x,y,facecolor='blue', alpha=0.2)
    if ylim:
        gca().set_ylim(ylim)
    

    try:
        HDI=sie.credible_interval(dist)
        yl=gca().get_ylim()
        text((HDI[0]+HDI[2])/2, 0.15*yl[1],'95% HDI', ha='center', va='center',fontsize=12)
        plot(HDI,[yl[1]*.1,yl[1]*.1,yl[1]*.1],'k.-',linewidth=1)
        for v in HDI:
            text(v, 0.05*yl[1],'%.3f' % v, ha='center', va='center', 
                 fontsize=12)
    except AttributeError:
        pass
    

def histogram(y,bins=50,plot=True):
    import numpy as np
    import pylab as py
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


    
def sampleplot_K(r,ylim=None,HDI_y=None):
    from pylab import plot,fill_between,gca,text
    
    x,y=histogram(r,plot=False)
    
    plot(x,y,'-o')
    
    fill_between(x,y,facecolor='blue', alpha=0.2)
    if ylim:
        gca().set_ylim(ylim)

    dx=x[1]-x[0]
    cs=np.cumsum(y)*dx
    
    HDI=np.percentile(r,[2.5,50,97.5])

    yl=gca().get_ylim()
    
    dy=0.05*yl[1]
    if HDI_y is None:
        HDI_y=yl[1]*.1
        
    
    text((HDI[0]+HDI[2])/2, HDI_y+dy,'95% HDI', ha='center', va='center',fontsize=12)
    plot(HDI,[HDI_y,HDI_y,HDI_y],'k.-',linewidth=1)
    for v in HDI:
        text(v, HDI_y-dy,'%.3f' % v, ha='center', va='center', 
             fontsize=12)
    xl=gca().get_xlim()
    
    text(.05*(xl[1]-xl[0])+xl[0], 0.9*yl[1],r'$\tilde{x}=%.3f$' % np.median(r), ha='left', va='center')
    

def distplot(var,label=None,
    show_quartiles=True,
    fill_between_quartiles=[],
    fill_between_values=[],
    xlim=None,
    values=None,
    show_percentile_on_values=False,
    fignum=None,
    figsize=None,
    quartiles=array([1,5,10,25,50,75,90,95,99])/100.0,
    quartile_format='{:.0%}',
    notebook=False,
    ):

    qmin=.0001
    qmax=1-qmin
    
    if xlim is None:
        xmin=var.ppf(qmin)
        xmax=var.ppf(qmax)
        xl=None
    else:
        xmin,xmax=xlim
        xl=[xmin,xmax]
    
    x=linspace(xmin,xmax,3000)
    dx=x[1]-x[0]
    
    y=var.pdf(x)
    
    yl=[-max(y)*.12, max(y)*1.25]
    
    
    
    if not notebook:
        if fignum is None:
            figure()
        else:
            figure(fignum)
            
        fig=gcf()
        if figsize is None:
            fig.set_size_inches( 13.75  ,   9.1375,forward=True)
        else:
            fig.set_size_inches(*figsize,forward=True)
    
        clf()
        show()
    
    fig=gcf()
    if figsize is None:
        fig.set_size_inches( 13.75  ,   9.1375,forward=True)
    else:
        fig.set_size_inches(*figsize,forward=True)

    plot(x,y,'-',linewidth=3)
    
    gca().set_ylim(yl)
    
    if xl is None:
        xl=gca().get_xlim()
    
    
    yl=gca().get_ylim()
    if show_quartiles:
        for j,q in enumerate(quartiles):
            xx=var.ppf(q)
            yy=var.pdf(xx)
            
            plot([xx,xx],yl,'g--')
            
            yt=(yl[1]-yl[0])*.8+yl[0]
            yt=yy+(yl[1]-yl[0])*.12
            text(xx,yt,quartile_format.format(q),horizontalalignment='center',color='g')
        
            yt=yl[0]/1.1-(j%2)*yl[0]/2.3
            text(xx,yt,'%.2f'% xx,horizontalalignment='center',color='b')
        
    plot(x,0*x,'k-')

    if fill_between_quartiles:        
        # fill
        q=fill_between_quartiles[0]
        xmin=var.ppf(q)

        q=fill_between_quartiles[1]
        xmax=var.ppf(q)
        
        xf=x[x>xmin]
        xf=xf[xf<xmax]
        yf=var.pdf(xf)
        
        xf=concatenate(((xf[0],),xf,(xf[-1],)))
        yf=concatenate(((0,),yf,(0,)))
        
        fill(xf,yf,facecolor='blue', alpha=0.2)    

    if fill_between_values:        
        # fill
        q=fill_between_values[0]
        idx=where(x<q)[0]
        try:    
            imin=idx[-1]
        except IndexError:
            imin=0
        
        q=fill_between_values[1]
        idx=where(x<q)[0]    
        imax=idx[-1]
    
        xf=concatenate(((x[imin],),x[imin:imax],(x[imax],)))
        yf=concatenate(((0,),y[imin:imax],(0,)))
        
        fill(xf,yf,facecolor='green', alpha=0.2)    
        
    if values:
        values=array(values)
        plot(values,0*values,'o',markersize=10)
        
        for v in values:
            ax=gca()
            ax.annotate('%.2f' % v, xy=(v, 0),  xycoords='data',
                        xytext=(50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"),
                        color=[0,.4,0],  # green
                        )
                        
                        
        transOffset = matplotlib.transforms.offset_copy(ax.transData, 
                            fig=fig,
                            x = -50, y=60, units='points')
                                                    
        if show_percentile_on_values:
            for v in values:
                text(v,0,'%.2f'% var.cdf(v),color='b',
                        transform=transOffset)
            


        
    if label:
        xlabel(label)
        s=label.split('(')[0].strip()
        ylabel('P(%s)' % s)
        
    gca().set_ylim(yl)
    gca().set_xlim(xl)
    if not notebook:
        draw()
    


def gaussplot(mu,sd,
    fill_between_quartiles=[],
    fill_between_values=[],
    values=None,
    ):
    
    Ns=5
    mnmx=[mu-Ns*sd,mu+Ns*sd]
    
    x=linspace(mnmx[0],mnmx[1],3000)
    dx=x[1]-x[0]
    
    y=exp(-(x-mu)**2/(2*sd**2))
    y=y/sum(y)/dx
    cy=cumsum(y*dx)
    
    yl=[-max(y)*.12, max(y)*1.25]
    
    quartiles=array([1,5,10,25,50,75,90,95,99])/100.0
    
    
    figure()
    fig=gcf()
    fig.set_size_inches( 13.75  ,   9.1375,forward=True)
    clf()
    show()
    
    plot(x,y,'-',linewidth=3)
    
    gca().set_ylim(yl)
    
    yl=gca().get_ylim()
    for j,q in enumerate(quartiles):
        idx=where(cy<q)[0]    
        i=idx[-1]
        xx=x[i]
        yy=y[i]
        plot([xx,xx],yl,'k:')
    
        yt=(yl[1]-yl[0])*.8+yl[0]
        yt=yy+(yl[1]-yl[0])*.12
        text(xx,yt,str(int(q*100))+'%',horizontalalignment='center')
    
        yt=yl[0]/1.1-(j%2)*yl[0]/2.3
        text(xx,yt,'%.2f'% xx,horizontalalignment='center')
    
        plot(x,0*x,'k-')
        
    if fill_between_quartiles:        
        # fill
        q=fill_between_quartiles[0]
        idx=where(cy<q)[0]    
        imin=idx[-1]
        
        q=fill_between_quartiles[1]
        idx=where(cy<q)[0]    
        imax=idx[-1]
    
        xf=concatenate(((x[imin],),x[imin:imax],(x[imax],)))
        yf=concatenate(((0,),y[imin:imax],(0,)))
        
        fill(xf,yf,facecolor='blue', alpha=0.2)    

    if fill_between_values:        
        # fill
        q=fill_between_values[0]
        idx=where(x<q)[0]    
        imin=idx[-1]
        
        q=fill_between_values[1]
        idx=where(x<q)[0]    
        imax=idx[-1]
    
        xf=concatenate(((x[imin],),x[imin:imax],(x[imax],)))
        yf=concatenate(((0,),y[imin:imax],(0,)))
        
        fill(xf,yf,facecolor='green', alpha=0.2)    
        
    if values:
        values=array(values)
        plot(values,0*values,'o',markersize=10)
        
        for v in values:
            ax=gca()
            ax.annotate('%.1f' % v, xy=(v, 0),  xycoords='data',
                        xytext=(-50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->")
                        )

        
    gca().set_ylim(yl)
    draw()
    
   
   
    
    
rc('font',size=size,family=family)
rc('axes',labelsize=size)
rc('axes',titlesize=size)
rc('xtick',labelsize=size)
rc('ytick',labelsize=size)
rc('legend',fontsize=size)

def nchoosek(N,k,exact=0):
    from scipy.misc import comb
    return comb(N,k,exact)
    
    
def binomial(p,h,N):
    dist=D.binom(N,p)
    return dist.pmf(h)

    
    
def load_data(fname):
    if ".csv" in fname:
        return pandas.read_csv(fname)
    elif ".xls" in fname:
        return pandas.read_excel(fname)
    else:
        raise ValueError("Not Implemented for this type of file (%s)" % fname)
        
def print_data(data):
    print (data.to_string())
    
def describe_data(data):
    print (data.describe().to_string())
    

def t_test(data,value):
    import scistats as stats
    import numpy as np
    
    t, tProb = stats.ttest_1samp(data, value)    
    
    
def countbins(min,max=None,step=1):
    if max is None:
        max=min
        min=0
        
    bins=arange(min,max+1.01,step)-0.5
    return bins
    
    
def distplot2(vars,label=None,
    show_quartiles=True,
    fill_between_quartiles=[],
    fill_between_values=[],
    xlim=None,
    values=None,
    fignum=None,
    figsize=None,
    quartiles=array([1,5,10,25,50,75,90,95,99])/100.0,
    notebook=False,
    ):

    qmin=.0001
    qmax=1-qmin
    
    if xlim is None:
        xmins=[var.ppf(qmin) for var in vars]
        xmaxs=[var.ppf(qmax) for var in vars]
        
        xmin=min(xmins)
        xmax=max(xmaxs)
        
        xl=None
    else:
        xmin,xmax=xlim
        xl=[xmin,xmax]
    
    x=linspace(xmin,xmax,3000)
    dx=x[1]-x[0]

    ys=[var.pdf(x) for var in vars]    

    maxys=[max(y) for y in ys]
        
    yl=[-max(maxys)*.12, max(maxys)*1.25]
    
    if not notebook:
        if fignum is None:
            figure()
        else:
            figure(fignum)
            
        fig=gcf()
        if figsize is None:
            fig.set_size_inches( 13.75  ,   9.1375,forward=True)
        else:
            fig.set_size_inches(*figsize,forward=True)
    
        clf()
        show()
    
    fig=gcf()
    if figsize is None:
        fig.set_size_inches( 13.75  ,   9.1375,forward=True)
    else:
        fig.set_size_inches(*figsize,forward=True)


    for y in ys:    
        plot(x,y,'-',linewidth=3)
    
    gca().set_ylim(yl)
    
    if xl is None:
        xl=gca().get_xlim()
    
    
    yl=gca().get_ylim()
    colors=['b','g','r','c','m','y','k']
    if show_quartiles:
        for idx,var in enumerate(vars):
            for j,q in enumerate(quartiles):
                xx=var.ppf(q)
                yy=var.pdf(xx)
                
                plot([xx,xx],yl,colors[idx]+':')
                
                yt=(yl[1]-yl[0])*.8+yl[0]
                yt=yy+(yl[1]-yl[0])*.12
                text(xx,yt,str(int(q*100))+'%',
                    horizontalalignment='center',
                    color=colors[idx])
            
                yt=yl[0]/1.1-(j%2)*yl[0]/2.3
                text(xx,yt,'%.2f'% xx,horizontalalignment='center',
                    color=colors[idx])
            
                plot(x,0*x,'k-')

    if fill_between_quartiles:        
        # fill
        q=fill_between_quartiles[0]
        xmin=var.ppf(q)

        q=fill_between_quartiles[1]
        xmax=var.ppf(q)
        
        xf=x[x>xmin]
        xf=xf[xf<xmax]
        yf=var.pdf(xf)
        
        xf=concatenate(((xf[0],),xf,(xf[-1],)))
        yf=concatenate(((0,),yf,(0,)))
        
        fill(xf,yf,facecolor='blue', alpha=0.2)    

    if fill_between_values:        
        # fill
        q=fill_between_values[0]
        idx=where(x<q)[0]
        try:    
            imin=idx[-1]
        except IndexError:
            imin=0
        
        q=fill_between_values[1]
        idx=where(x<q)[0]    
        imax=idx[-1]
    
        xf=concatenate(((x[imin],),x[imin:imax],(x[imax],)))
        yf=concatenate(((0,),y[imin:imax],(0,)))
        
        fill(xf,yf,facecolor='green', alpha=0.2)    
        
    if values:
        values=array(values)
        plot(values,0*values,'o',markersize=10)
        
        for v in values:
            ax=gca()
            ax.annotate('%.1f' % v, xy=(v, 0),  xycoords='data',
                        xytext=(-50, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"),
                        color=[0,.4,0],  # green
                        )

        
    if label:
        xlabel(label)
        s=label.split('(')[0].strip()
        ylabel('P(%s)' % s)
        
    gca().set_ylim(yl)
    gca().set_xlim(xl)

    if not notebook:
        draw()
  


def remove_nan(x,y):
    try:
        x=x[y.notnull()]
        y=y[y.notnull()]
    except AttributeError:
        x=x[~isnan(y)]
        y=y[~isnan(y)]
        
    return x,y
    
    
    
def fit(x,y,funcstr,*args,**kwargs):

    x=pandas.Series(array(x))
    y=pandas.Series(array(y))

    x,y=remove_nan(x,y)
    
    
    if funcstr=='linear':
        result=fit(x,y,'power',1)
        result.type='linear'
    elif funcstr=='quadratic':
        result=fit(x,y,'power',2)
        result.type='quadratic'
    elif funcstr=='exponential':
        y2=np.log(y)
        result=fit(x,y2,'linear')
        result.params=[np.exp(result.params[1]),result.params[0]]
        p=result.params
        labelstr='y= %.4e exp(%.4e x)' % (p[0],p[1])
        result.label=labelstr
        result.type='exponential'
    
    elif funcstr=='power':
        data=pandas.DataFrame({'x':x,'y':y})
        power=args[0]
        
        keys=['x']
        for i in range(power-1):
            exponent=(i+2)
            key='x%d' % exponent
            data[key] = x**exponent
            keys.append(key)

        result2=pandas.ols(y=data['y'],x=data[keys])
        keys.reverse()
        keys+=['intercept']
        
        p=[result2.beta[s] for s in keys]

        labelstr='y= '
        for i,pv in enumerate(p):
            pw=len(p)-i-1
            if pw==1:
                labelstr+='%.4e x + ' % (pv)
            elif pw==0:
                labelstr+='%.4e + ' % (pv)
            else:
                labelstr+='%.4e x^%d + ' % (pv,pw)
        labelstr=labelstr[:-3]  # take off the last +
        
        
        result=Struct()
        result.params=p
        result.type='power'
        result.label=labelstr   
        result.pandas_result=result2
        
    else:
        raise ValueError('Unknown fit name %s' % funcstr)
        
    return result
        
def fitval(result,x):
    x=pandas.Series(array(x))

    if result.type=='linear':
        y=result.params[0]*x+result.params[1]
    elif result.type=='quadratic':
        y=result.params[0]*x**2+result.params[1]*x+result.params[2]
    elif result.type=='power':
        y=0.0
        for i,pv in enumerate(result.params):
            pw=len(result.params)-i-1
            y+=pv*x**pw
    elif result.type=='exponential':
        y=result.params[0]*np.exp(x*result.params[1])
    else:
        raise ValueError('Unknown fit name %s' % result.type)
        
    return y
    

def _predict(res):

    def fun(**kwargs):
        return res.predict(exog=kwargs)

    return fun

def regression(equation,data,plot=True):
    import statsmodels.formula.api as smf
    import warnings    

    # supress the low-N kurtosis test warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.ols(formula=equation, data=data)
        res = mod.fit()

    df=res.df_resid
    result=Struct()

    result['_Normal_Approximation']={}
    result['_Predict']=_predict(res)
    for key in res.params.keys():
        result[key]=tdist(df,res.params[key],res.bse[key])

        N=res.nobs
        k=1+20.0/N**2
        result['_Normal_Approximation'][key]=[res.params[key],res.bse[key]*k]

    result['_Fit_Results']=res
    result['_Cov']=res.cov_params()
    result['_R2']=res.rsquared
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result['_Summary']=res.summary()


    if plot:
        for key in res.params.keys():
            var=result[key]
            xmin=var.ppf(0.001)
            xmax=var.ppf(0.999)
            if key=="Intercept":
                label=key
            else:
                key=key.replace("_",' ')
                label=r'$\beta_{\rm %s}$' % key
            
            distplot(var,
                xlim=[xmin,xmax],
                label=label,
                figsize=(16,8),
                )

    return result



 