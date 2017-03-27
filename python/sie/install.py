#!/usr/bin/env python
from __future__ import print_function
import os,sys

if not os.path.exists('setup.py'):
    print ("An error occurred trying to install.  This is likely to be because you did")
    print ("not actually *extract* the .zip file, and are trying to run this install.py")
    print ("file inside the zip by double-clicking.")
    print()
    print ("Please extract the .zip file, and rerun this install.py.")
    print()
    x=raw_input("<hit enter to continue>")
    raise ValueError
    
os.system('python setup.py install')
os.system('pip install emcee')

