from __future__ import division
from plotty import *
from utils import fact,Fibn
import itertools

savefigs=[3]

for fig in savefigs:
    if fig==1:
        N=30
        p=[]
        for h in range(31):
        
            p.append(fact(N)/fact(h)/fact(N-h) * (0.5)**N)
            
            
        h=range(31)
        
        figure(1)
        #lineplot(h,p)
        plot(h,p,'--o',linewidth=3,markersize=10)
        xlabel('Number of heads')
        ylabel('$P(h,N=30)$')
        grid(True)
        draw()
        
        savefig('../../figs/coinflips1.pdf')

    elif fig==2:
        # solution for streaks http://marknelson.us/2011/01/17/20-heads-in-a-row-what-are-the-odds/
        N=50
        streak=9
        
        if False:
            
            ss='H'*streak
            
            total=0
            count=0
            for v in itertools.product(['H','T'],repeat=N):
                s=''.join(v)
                if ss in s:
                    count+=1
                total+=1
                
            p=float(count)/total
            print p
            
                
        import random
        
        ss='H'*streak
        ss2='T'*streak
        total=0
        count=0
        while True:
            for i in range(100):
                s=''.join([random.choice(['H','T']) for _ in range(N)])
                if ss in s or ss2 in s:
                    count+=1
                total+=1
            
            p=float(count)/total
            print p
        
        
    elif fig==3:
        N=30
        figure(1)
        for pp in [0.1,0.5,0.8]:
            p=[]
            for h in range(31):
            
                p.append(fact(N)/fact(h)/fact(N-h) * (pp)**h*(1-pp)**(N-h))
                
                
            h=range(31)
            
            
            
            #lineplot(h,p)
            plot(h,p,'--o',linewidth=3,markersize=10,label='$p=%.1f$' % pp)
            xlabel('Number of heads')
            ylabel('$P(h,N=30)$')
            grid(True)
        legend(loc=0)
        draw()
        
        savefig('../../figs/coinflips5.pdf')
        