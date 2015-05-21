#!/usr/bin/env python
import os

value=os.system("ipython notebook --pylab inline")

if value>0:
    value=os.system("ipython notebook --pylab inline --port 9999")

