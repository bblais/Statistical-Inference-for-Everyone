from .sie import *
import sie
import sie.mcmc as mcmc

import matplotlib.pyplot as plt

# there really should be a better way than this
path,junk=os.path.split(sie.__file__)
plt.style.use(path+'/sie.mplstyle')

__version__="0.0.7"
