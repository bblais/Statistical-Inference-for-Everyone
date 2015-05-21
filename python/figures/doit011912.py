from pylab import *
from numpy import *
from bigfonts import *
mu=5
sd=3

Ns=5
mnmx=[mu-Ns*sd,mu+Ns*sd]

x=linspace(mnmx[0],mnmx[1],3000)
dx=x[1]-x[0]

y=exp(-(x-mu)**2/(2*sd**2))
y=y/sum(y)/dx
cy=cumsum(y*dx)

yl=[-max(y)*.12, max(y)*1.25]

quartiles=array([1,5,10,25,50,75,90,95,99])/100.0


figure(1)
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
    
# fill
q=0.1
idx=where(cy<q)[0]    
imin=idx[-1]

q=0.9
idx=where(cy<q)[0]    
imax=idx[-1]

xf=concatenate(((x[imin],),x[imin:imax],(x[imax],)))
yf=concatenate(((0,),y[imin:imax],(0,)))

fill(xf,yf,facecolor='blue', alpha=0.2)    
    
gca().set_ylim(yl)
draw()