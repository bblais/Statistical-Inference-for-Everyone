from utils import *


x=0.5

h=5
N=8

c=coinflip(h,N)
xvals=linspace(0,1,1000)
dx=xvals[1]-xvals[0]

K=sum(c.pdf(xvals)*dx)

p_x=2.0 # normalization constant for uniform x from 0.5 to 1


p=0.0
for x in xvals:
    p+=c.pdf(x)*p_x*dx/K
    
print p

