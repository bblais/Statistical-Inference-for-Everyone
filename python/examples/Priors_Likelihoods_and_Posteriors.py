# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sie import *

# <markdowncell>

# ## Estimating Lengths
# 
# ### Known deviation, $\sigma$

# <codecell>

x=[5.1, 4.9, 4.7, 4.9, 5.0]
sigma=0.5

# <codecell>

mu=sample_mean(x)
N=len(x)

# <codecell>

dist=normal(mu,sigma/sqrt(N))
distplot(dist)

# <codecell>

credible_interval(dist)

# <markdowncell>

# ### Unknown $\sigma$

# <codecell>

mu=sample_mean(x)
s=sample_deviation(x)
print mu,s

# <codecell>

dist=tdist(N-1,mu,s/sqrt(N))

# <codecell>

distplot(dist,xlim=[4.6,5.4])

# <codecell>

credible_interval(dist)

# <codecell>


