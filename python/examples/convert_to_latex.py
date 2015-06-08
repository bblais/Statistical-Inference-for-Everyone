#!/usr/bin/env python

import os, sys

if len(sys.argv)==1:
    print "No files given."

for fname in sys.argv[1:]:
    cmd="ipython nbconvert --to latex %s --template bblais" % fname
    print cmd
    os.system(cmd)

