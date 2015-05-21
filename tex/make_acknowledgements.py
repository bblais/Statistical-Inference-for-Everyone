#!/usr/bin/python
import os

with open('acknowledgements_list.txt') as fid:
    lines=fid.readlines()

people=[x.split() for x in lines if x]
people=[ (x[1],x[0]) for x in people]
people=sorted(people)

num_cols=3
with open('acknowledgements.tex','w') as fid:
    fid.write(r"""
    \chapter*{Acknowledgements}
    \thispagestyle{empty}

    I would like to acknowledge the following people who have added to this book, in big ways and in small.  The book is much better as a result. 

    \vspace{.5in}

    \begin{doublespace}
    \begin{tabular}{%s}
    """ % ("p{2in}"*num_cols))

    for i,person in enumerate(people):
        fid.write(person[1]+" "+person[0])
        if (i+1)%num_cols==0:
            fid.write(r'\\'+"\n")
        else:
            fid.write(r'&')
    fid.write(r"""
     \end{tabular}
    \end{doublespace}
    """)

os.system('cat acknowledgements.tex')
