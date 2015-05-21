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
    
    
def floatstr(v,decimals=3,max=12):

    if v==0.0:
        return "0."+"0"*decimals
        
    s1='%e' % v
    parts=s1.split('e')
    num=parts[0].replace('.','')
    
    s=''
    if parts[1][0]=='-':  # small decimal
        power=int(parts[1][1:])
        s+='0.'
        s+='0'*(power-1)
        s+=num[:decimals]
        return s
    else:  # large
        return '%f' % v
        
pvals=linspace(0,1,11)

probs=[binomial(p,3,12)*(1.0/11) for p in pvals]

K=sum(probs)

print r"\begin{tabular}{ccc}"
print r"Model & $\sim P(M_i|{\rm data}=\{9T,3H\})$ & $\sim P(M_i|{\rm data}=\{9T,3H\})/K$ \\\hline\hline"
for i,p in enumerate(pvals):
    v=binomial(p,3,12)*1.0/11
    print r"$M_{%d}$ & %s & %s\\" % (i,floatstr(v),floatstr(v/K))
    
print r"\cline{2-2}&$K$=%s & " % (floatstr(K))
    
print r"\end{tabular}"


p=probs/K

figure(1)
bigfonts(30)
#lineplot(range(len(p)),p,markersize=10)
plot(p,'o--',markersize=10,linewidth=3)
ax=gca()
ax.set_xticks(range(11))
ax.set_xlim([-.5,10.5])
grid(True)
xlabel('Model Number')
ylabel('P(model|data={9T,3H})')
draw()
savefig('../../figs/bentcoinprobs1.pdf')



s="%s" % floatstr(p[0])
for i in range(1,5):
    s=s+"+%s" % floatstr(p[i])
s=s+'=%s' % floatstr(sum(p[:5]))
print s


data="T T T H T H T T T T T H"
data=data.replace(" ","")

for fig in [2,3,4]:
    i=(fig-2)*3


    figure(fig,figsize=(18,6))
    bigfonts(23)
    
    ymax={2:.3501,3:.3501,4:.3501,5:.3501}
    
    if fig in ymax:
        ym=ymax[fig]
    else:
        ym=None
    
    subplot(1,3,1)
    
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
    #lineplot(range(len(p)),p,markersize=10)
    plot(p,'o--',markersize=10,linewidth=3)
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    ylabel('P(model|data)')
    title('data={%s}' % subdata)
    
    subplot(1,3,2)
    i=i+1
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
#    lineplot(range(len(p)),p,markersize=10)
    plot(p,'o--',markersize=10,linewidth=3)
    #plot(p,'o-')
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    title('data={%s}' % subdata)
    
    subplot(1,3,3)
    i=i+1
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
    #lineplot(range(len(p)),p,markersize=10)
    plot(p,'o--',markersize=10,linewidth=3)
    #plot(p,'o-')
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    title('data={%s}' % subdata)
    
    
    
    draw()
    savefig('../../figs/bentcoinprobs%d.pdf' % fig)
    
    
#=================
fig=5
if True:
    i=0


    figure(fig,figsize=(18,6))
    
    ymax={2:.3501,3:.3501,4:.3501,5:.3501}
    
    if fig in ymax:
        ym=ymax[fig]
    else:
        ym=None
    
    subplot(1,3,1)
    
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
    #lineplot(range(len(p)),p,markersize=10)
    #plot(p,'o-')
    plot(p,'o--',markersize=10,linewidth=3)
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    ylabel('P(model|data)')
    title('data={%s}' % subdata)
    
    subplot(1,3,2)
    i=6
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
    #lineplot(range(len(p)),p,markersize=10)
    #plot(p,'o-')
    plot(p,'o--',markersize=10,linewidth=3)
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    title('data={%s}' % subdata)
    
    subplot(1,3,3)
    i=13
    subdata=data[:i]
    H,T=subdata.count("H"),subdata.count("T")
    N=H+T
    
    probs=[binomial(p,H,N)*1.0/11.0 for p in pvals]
    K=sum(probs)
    p=probs/K
    
    #lineplot(range(len(p)),p,markersize=10)
    #plot(p,'o-')
    plot(p,'o--',markersize=10,linewidth=3)
    ax=gca()
    ax.set_xticks(range(11))
    ax.set_xlim([-.5,10.5])
    yl=ax.get_ylim()
    if ym:
        ax.set_ylim([0,ym])
    else:
        ax.set_ylim([0,yl[1]])
    grid(True)
    xlabel('Model Number')
    title('data={%s}' % subdata)
    
    
    
    draw()    
    savefig('../../figs/bentcoinprobs%d.pdf' % fig)
    