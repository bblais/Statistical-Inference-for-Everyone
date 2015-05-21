# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sie import *

# <markdowncell>

# ## Beta Distribution Example
# 
# ### 3 heads and 9 tails

# <markdowncell>

# Plot a beta distribution with 3 heads and 9 tails...

# <codecell>

dist=beta(h=1,N=3)
distplot(dist,xlim=[0,1],show_quartiles=False)

# <markdowncell>

# The median of this distribution...

# <codecell>

dist.median()

# <markdowncell>

# the 95% credible interval, with the median in the middle,

# <codecell>

credible_interval(dist)

# <markdowncell>

# ### 1 heads and 3 tails
# 
# This should be about the same fraction as the previous example, but broader

# <codecell>

dist=beta(h=1,N=4)
distplot(dist,xlim=[0,1])

# <codecell>

credible_interval(dist)

# <codecell>


