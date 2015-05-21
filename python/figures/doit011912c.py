from utils import *

var=gaussian(6.5,1.7/sqrt(85))
c=0.95
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='time before promotion (years)')
    
title('Problem #1 (normal)')

var=tdist(24,14.381,1.892/sqrt(25))
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='loan (thousand dollars)')
    
title('Problem #2 (t-dist)')

var=coinflip(350,500)
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='proportion of votors')
    
title('Problem #3 (beta)')

var=gaussian(45.5,3/sqrt(120))
c=0.95
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='income (thousand dollars)',
    values=[45.0])
title('Problem #6 (normal)')


c=0.995
var=tdist(9,1.3,0.9/sqrt(25))
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='copies (millions)',
    values=[1.5])
    
title('Problem #7 (t-dist)')

c=0.98
var=coinflip(140,200)
distplot(var,fill_between_quartiles=[(1-c)/2,1-(1-c)/2],
    label='proportion of arrears',
    values=[0.6])
    
title('Problem #8 (beta)')


draw()

for fig in get_fignums():
    figure(fig)
    savefig('fig011912_%d.png' % fig)
    