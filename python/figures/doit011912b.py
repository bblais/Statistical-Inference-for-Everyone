from utils import *

var=gaussian(5,3)

distplot(var)

distplot(var,fill_between_quartiles=[0.1,0.9])

distplot(var,fill_between_values=[2,6])

distplot(var,fill_between_quartiles=[0.1,0.9],
    values=[3.6,5.2,9.3],
    label='this (dollars)')
