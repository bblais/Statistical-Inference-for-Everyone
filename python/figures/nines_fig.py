from __future__ import division
from plotty import *


figure(1)
bigfonts(30)

m=arange(0,15)

pN=1**m * 1e-6
pL=(2/55)**m * 1/2
pH=(9/55)**m * 1/2

K=pN+pL+pH

pN=pN/K
pL=pL/K
pH=pH/K

if False:
    marker=['*','o','s']
    color=['b','g','r']
    for mk,cl,p in zip(marker,color,[pL,pH,pN]):
        if mk=='*':
            ms=15
        else:
            ms=10
        
        plot(m,p,marker=mk,color=cl,
            linewidth=2,markersize=ms,linestyle='dashed', )
    legend(['Low Deck','High Deck','Nines Deck'],0)
    xlabel("Number of 9's in Drawn in a Row")
    ylabel('Model Posterior Probability')
    grid(True)
    savefig('../../figs/nines_HLN.pdf')
    draw()

if True:
    marker=['*','o','s']
    color=['b','g','r']
    for mk,cl,p in zip(marker,color,[pL,pH,pN]):
        if mk=='*':
            ms=15
        else:
            ms=10
        
        plot(m,p,marker=mk,color=cl,
            linewidth=3,markersize=ms,linestyle='dashed')
    legend(['Low Deck','High Deck','Nines Deck'],0)
    xlabel("Number of 9's in Drawn in a Row")
    ylabel('Model Posterior Probability')
    grid(True)
    savefig('../../figs/nines_HLN.pdf')
    draw()
