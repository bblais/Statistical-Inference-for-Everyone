#!/usr/bin/env python

# <nbformat>2</nbformat>

# <codecell>
from glob import glob
import sys, os
from IPython.nbformat import current
from IPython.utils.text import wrap_paragraphs

# <codecell>


def new_figure(data, fmt):
    """Create a new figure file in the given format.

    Returns a path relative to the input file.
    """
    figname = '%s_fig_%02i.%s' % (self.infile_root,
                                  self.figures_counter, fmt)
    self.figures_counter += 1
    fullname = os.path.join(self.files_dir, figname)

    # Binary files are base64-encoded, SVG is already XML
    if fmt in ('png', 'jpg', 'pdf'):
        data = data.decode('base64')
        fopen = lambda fname: open(fname, 'wb')
    else:
        fopen = lambda fname: codecs.open(fname, 'wb', self.default_encoding)

    with fopen(fullname) as f:
        f.write(data)

    return fullname

def export_latex(fname):
    with open(fname) as f:
        nb = current.read(f, 'json')
        
    base,ext=os.path.splitext(fname)
    figdirname=base
    if not os.path.exists(figdirname):
        os.mkdir(figdirname)
        
    lines = ''
    
    figcount=0
    figname,ext=os.path.splitext(fname)
    for cell in nb.worksheets[0].cells:
        if cell.cell_type == u'code':
            lines += '\\begin{lstlisting}\n'
            lines += '%s\n' % cell.input
            lines += '\\end{lstlisting}\n'
            for output in cell.outputs:
                if output.output_type == u'pyout':
                    if hasattr(output, 'latex'):
                        s=output.latex
                        s=s.replace(r"\\[",r"\[")
                        s=s.replace(r"\\]",r"\]")
                        lines += '%s\n' % s
                    else:
                        lines += '\n'
                        lines += '\\begin{verbatim}\n'
                        lines += '%s\n' % output.text
                        lines += '\\end{verbatim}\n'
                elif str(output.output_type) == 'display_data':
                    if 'png' in output:
                        pic=output['png'].decode('base64')
                        fig_fname='%s/%s_fig%d.png' % (figdirname,figname,figcount)
                        fopen = lambda fname: open(fname, 'wb')
                        print "Writing %s..." % fig_fname
                        with fopen(fig_fname) as f:
                            f.write(pic)

                        lines += '\n'
#                         lines += '(see Figure~\\ref{%s_%d})\n' % (figname,figcount)
#                         lines += '\\begin{figure*}\n'
                        lines += '\\begin{center}\\includegraphics[width=4.5in]{%s}\\end{center}\n' % (fig_fname)
#                         lines += '\\label{%s_%d}\n' % (figname,figcount)
#                         lines += '\\end{figure*}\n'
                        
                        figcount+=1
                    elif 'text' in output:
                        lines += '\n'
                        lines += '\\begin{verbatim}\n'
                        lines += '%s' % output.text
                        lines += '\\end{verbatim}\n'                    
                        
                    else:
                        print "Unknown output type",output.output_type
                        print output
                        print "input",cell.input
                    
                elif output.output_type == u'stream':
                    lines += '\n'
                    lines += '\\begin{verbatim}\n'
                    lines += '%s' % output.text
                    lines += '\\end{verbatim}\n'                    
                else:
                    print "Unknown output type",output.output_type
                    print type(output.output_type)
                    print output
                    print "input",cell.input
                
            lines += '\n'
        elif cell.cell_type == u'markdown':
            paragraphs = wrap_paragraphs(cell.source)
            for p in paragraphs:
                s=p
                s=s.replace(r"\\[",r"\[")
                s=s.replace(r"\\]",r"\]")
            
                if s.startswith('## '):
                    s=r"\subsection{%s}" % (s[3:].strip())
                    s+="\n"
                if s.startswith('### '):
                    s=r"\subsubsection{%s}" % (s[3:].strip())
                    s+="\n"
            
                lines += s
                lines += '\n\n'
        else:
            print "Unknown cell type",cell.cell_type
            
    newfname = os.path.splitext(fname)[0] + '.tex'
    with open(newfname,'w') as f:
        f.write(lines.encode('utf8'))

# <codecell>

if __name__ == '__main__':
    for f in sys.argv[1:]:
        fnames=glob(f)
        for fname in fnames:
            export_latex(fname)
