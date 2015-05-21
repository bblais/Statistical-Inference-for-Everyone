from plotty import *

    
    
#savefigs=[1,2,3,4]
savefigs=[2,4]

for fig in savefigs:

    if fig==1:
        figure(1,(6,6))
        
        p=array([1,2,3,4,5,6,5,4,3,2,1])
        sum=array([2,3,4,5,6,7,8,9,10,11,12])-0.4
        bigfonts(26)
        bar(sum,p)
        
        gca().set_xlim([1.5,12.5])
        xlabel('Sum of Two Dice')
        ylabel('Arrangements of the \nSum of Two Dice\n\n',horizontalalignment='center')
        gca().set_ylim([0,6.5])
        grid(True)
        gca().set_xticks([2,3,4,5,6,7,8,9,10,11,12])
        subplots_adjust(bottom=0.2,left=0.2) 
        draw()
        savefig('../../figs/sumdice1.pdf')

    elif fig==2:
        figure(2)
        bigfonts(36)
        
        p=array([1,2,3,4,5,6,5,4,3,2,1])/36.0
        sum=array([2,3,4,5,6,7,8,9,10,11,12])
        
        #lineplot(sum,p,markersize=15)
        plot(sum,p,'--o',linewidth=3,markersize=15)
        grid(True)
        gca().set_xlim([1.5,12.5])
        gca().set_ylim([0,0.18])
        xlabel('Sum of Two Dice')
        ylabel('$P($Sum of Two Dice$)$')
        
        adjustaxes([0.05,0.02])
        draw()
        savefig('../../figs/sumdice2.pdf')

    elif fig==3:    
        figure(3)
        
        N=20
        
        p=r_[arange(1,N+1),arange(N-1,0,-1)]
        sum=arange(2,2*N+1)-0.4
        
        bar(sum,p)
        
        gca().set_xlim([1.5,(2*N)+.5])
        xlabel('Sum of Two 20-Sided-Dice')
        ylabel('Arrangements of the Sum of Two 20-Sided-Dice')
        gca().set_ylim([0,20.5])
        grid(True)
        #gca().set_xticks(arange(2,2*N+1))
        draw()
        savefig('../../figs/sumdice3.pdf')

    elif fig==4:    
        figure(4)
        bigfonts(36)
        N=20
        p=r_[arange(1,N+1),arange(N-1,0,-1)]/float(N**2)
        sum=arange(2,2*N+1)
        
        #lineplot(sum,p,markersize=15)
        plot(sum,p,'--o',linewidth=3,markersize=15)
        
        grid(True)
        gca().set_xlim([1.5,(2*N)+.5])
        gca().set_ylim([0,.055])
        xlabel('Sum of Two 20-Sided-Dice')
        ylabel('$P($Sum of Two 20-Sided-Dice$)$')
        adjustaxes([0.05,0.02])
        draw()
        savefig('../../figs/sumdice4.pdf')

    else:
        raise ValueError,fig
        