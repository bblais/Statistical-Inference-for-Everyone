# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Histograms

# <codecell>

from sie import *

# <markdowncell>

# Load a sample data set, and select only the Male data...

# <codecell>

data=load_data('data/survey.csv')
male_data=data[data['Sex']=='Male']

# <markdowncell>

# select only the height data, and drop the missing data (na)...

# <codecell>

male_height=male_data['Height'].dropna()

# <markdowncell>

# make the histogram

# <codecell>

hist(male_height,bins=20)
xlabel('Height [cm]')
ylabel('Number of People')

# <markdowncell>

# ## Scatter Plot

# <codecell>

from sie import *

# <markdowncell>

# Load a sample data set, and select only the Male data...

# <codecell>

data=load_data('data/survey.csv')
male_data=data[data['Sex']=='Male']

# <markdowncell>

# select only the height and the width of writing hand data, and drop the missing data (na)...

# <codecell>

subdata=male_data[['Height','Wr.Hnd']].dropna()
height=subdata['Height']
wr_hand=subdata['Wr.Hnd']

# <markdowncell>

# plot the data

# <codecell>

plot(height,wr_hand,'o')
ylabel('Writing Hand Span [cm]')
xlabel('Height [cm]')

