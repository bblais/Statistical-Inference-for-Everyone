from plotty import *

def fact(N):
    if N==0:
        return 1
    else:
        return prod(xrange(1,N+1))

def nchoosek(N,k):
    
    return fact(N)//(fact(k)*fact(N-k))
    
    
    
def binomial(p,h,N):
    
    return nchoosek(N,h)*p**(h)*(1-p)**(N-h)



bigfonts(30)

x=linspace(0,1,1000)
y=ones(x.shape)

figure(1)
clf()
fig=gcf()
fig.set_size_inches(10,8,forward=True)
plot(x,y,'-',linewidth=3)
xmin=0
xmax=1

xf=x[x>xmin]
xf=xf[xf<xmax]
yf=ones(xf.shape)
xf=concatenate(((xf[0],),xf,(xf[-1],)))
yf=concatenate(((0,),yf,(0,)))

fill(xf,yf,facecolor='blue', alpha=0.2)    

xlabel(r'$\theta$')
ylabel(r'$P(\theta)$')

grid(True)
ax=gca()
ax.set_ylim([-.05,1.1])
text(0.5,0.5,'Area Under Curve = 1',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=45,fontsize=30,
         )
      
         
draw()
savefig('../../figs/bentcoindist1.pdf')

#

y=binomial(x,3,12)

figure(2)
clf()
fig=gcf()
fig.set_size_inches(12,9,forward=True)
plot(x,y,'-',linewidth=3)
xmin=0
xmax=1

xf=x[x>xmin]
xf=xf[xf<xmax]

yf=binomial(xf,3,12)
xf=concatenate(((xf[0],),xf,(xf[-1],)))
yf=concatenate(((0,),yf,(0,)))

fill(xf,yf,facecolor='blue', alpha=0.2)    

xlabel(r'$\theta$')
ylabel(r'$\sim P(\theta|{\rm data}=\{9T,3H\})$')
grid(True)

ax=gca()
ax.set_ylim([-.01,0.27])
text(0.3,0.1,'Area Under Curve NOT 1',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=45,fontsize=30,
         )
      
         
draw()
savefig('../../figs/bentcoindist2.pdf')

dx=x[1]-x[0]
K=sum(y*dx)
y=y/K
figure(3)
clf()
fig=gcf()
fig.set_size_inches(12,9,forward=True)
plot(x,y,'-',linewidth=3)
xmin=0
xmax=1

xf=x[x>xmin]
xf=xf[xf<xmax]

yf=binomial(xf,3,12)
yf=yf/K
xf=concatenate(((xf[0],),xf,(xf[-1],)))
yf=concatenate(((0,),yf,(0,)))

fill(xf,yf,facecolor='blue', alpha=0.2)    

xlabel(r'$\theta$')
ylabel(r'$P(\theta|{\rm data}=\{9T,3H\})$')
grid(True)

ax=gca()
ax.set_ylim([-.15,3.5])
text(0.22r,.9,'Area Under Curve=1',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=45,fontsize=30,
         )
      
         
draw()
savefig('../../figs/bentcoindist3.pdf')
