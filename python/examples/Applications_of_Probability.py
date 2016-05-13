
# coding: utf-8

# ## Coin Flips

# In[12]:

from sie import *


# Generate a small list of data...

# In[13]:

data=randint(2,size=10)
print data


# Generate a slightly larger list of data...

# In[14]:

data=randint(2,size=30)
print data


# In[15]:

data=randint(2,size=(2000,10))
data


# We have here a large collection of numbers (20000 of them!), organized in 2000 rows of 10 columns.  We can sum all of the 20000 values, or we can sum across columns or across rows, depending on what we want.

# In[16]:

sum(data)  # add up all of the 1's


# In[17]:

sum(data,axis=0)  # sum up all of the columns


# In[18]:

sum(data,axis=1)  # sum up all of the rows


# Typically the hist command makes its own bins, which may not center on the actual count values.  That's why we call countbins(N), to make bins centered on the counts.

# In[19]:

N=sum(data,axis=1)  # number of heads in each of many flips
hist(N,countbins(10))
xlabel('Number of Heads')
ylabel('Number of Flips')


# To get a probability distribution, we divide the histogram result by $N$.

# This distribution is Bernoulli's equation, or in other words, the binomial distribution.
# 
# \\[
# p(h,10) = {10 \choose h} 0.5^h \cdot 0.5 ^{10-h}
# \\]

# In[20]:

h=array([0,1,2,3,4,5,6,7,8,9,10])

# or...

h=arange(0,11)


# (recall that ** is exponentiation in Python, because the caret (\^{}) was already used for a computer-sciency role.)  The spaces in the equation below are not needed, but highlight the three parts of the binomial distribution.

# In[21]:

p=nchoosek(10,h)* 0.5**h * 0.5**(10-h)


# In[22]:

hist(N,countbins(10),normed=True)
plot(h,p,'--o')
xlabel('Number of Heads, $h$')
ylabel('$p(h|N=10)$')

