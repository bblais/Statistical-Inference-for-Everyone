from utils import *

gaussplot(5,3)

gaussplot(5,3,fill_between_quartiles=[0.1,0.9])

gaussplot(5,3,fill_between_values=[2,6])

gaussplot(5,3,fill_between_quartiles=[0.1,0.9],
    values=[3.6,5.2,9.3])
