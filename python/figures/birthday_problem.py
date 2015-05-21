from __future__ import print_function
from plotty import *

def prob(N):
    p=1
    for i in range(365-N,365):
        p*=float(i)/365
        
    return 1-p
    
    
    
people=range(2,91)
p=[]
for N in people:
    p.append(prob(N))
    
figure(1)
bigfonts(30)

lineplot(people,p)
#plot(people,p,'-o')
xlabel('Number of People')
ylabel('P(at least 2 with same birthday)')
grid(True)

plot([23,23],[0,prob(23)],'k-',linewidth=4)
plot([0,23],[prob(23),prob(23)],'k-',linewidth=4)
print(prob(23))
text(6,.7,'50% mark \nreached at\n23 people')
savefig('../../figs/birthday.pdf')
draw()
