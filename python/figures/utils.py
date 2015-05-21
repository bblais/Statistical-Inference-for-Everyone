from pylab import *
from numpy import *
from bigfonts import *
from scipy import stats
from scipy.stats import distributions as D

def uniform(min=0,max=1):
    return D.uniform(min,max-min)

def gaussian(mu=0,sd=1):
    return D.norm(mu,sd)


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

def distplot(var,label=None,
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
            text(xx,yt,str(int(q*100))+'%',horizontalalignment='center',color='g')
        
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

        
    if label:
        xlabel(label)
        s=label.split('(')[0].strip()
        ylabel('P(%s)' % s)
        
    gca().set_ylim(yl)
    gca().set_xlim(xl)
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
    
   
   
    
def fact(N):
    if N==0:
        return 1
    else:
        p=1
        for v in range(1,N+1):
            p=p*v
        return p

def nchoosek(N,k):
    
    return fact(N)//(fact(k)*fact(N-k))
    
    
def binomial(p,h,N):
    
    return nchoosek(N,h)*p**(h)*(1-p)**(N-h)

    
# http://mathworld.wolfram.com/Fibonaccin-StepNumber.html
def Fibn(k,n):
    L=[1]
    k=k-1
    for i in range(n-1):
        start=i-k
        if start<0:
            start=0
        s=sum(L[start:])
        L.append(s)
        
    return L

    
def distplot2(vars,label=None,
    show_quartiles=True,
    fill_between_quartiles=[],
    fill_between_values=[],
    xlim=None,
    values=None,
    fignum=None,
    figsize=None,
    quartiles=array([1,5,10,25,50,75,90,95,99])/100.0,
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
    draw()
  