"""Utilities to generate PDF plots (via Asymptote)."""

import logging
import subprocess

import pandas

from dataclasses import dataclass
from pathlib import Path
from perflib.utils import sjoin, cjoin
from typing import List
import tempfile
import numpy as np

import perflib.utils

top = Path(__file__).resolve().parent.parent


@dataclass
class BaseFigure:
    tag: str
    title: str
    caption: str
    docdir: Path
    labels: List[str]
    primary: List[Path]
    secondary: List[Path]
    figtype: str


class PDFFigure(BaseFigure):

    def asycmd(self):
        asycmd = ['asy', '-f', 'pdf']

        ndata = 0
        for filename in self.primary:
            df = pandas.read_csv(filename, sep="\t", comment='#')
            ndata = max(ndata, len(df.index))
        
        if ndata > 1 and self.figtype == "linegraph":
            asycmd.append(top / "datagraphs.asy")
        elif ndata == 1 or self.figtype == "bargraph":
            asycmd.append(top / "bargraph.asy")

        primary = [x.resolve() for x in self.primary]
        asycmd.extend(['-u', f'filenames="{cjoin(primary)}"'])

        if self.labels is not None:
            asycmd.extend(['-u', f'legendlist="{cjoin(self.labels)}"'])

        if self.secondary is not None:
            secondary = [x.resolve() for x in self.secondary]
            asycmd.extend(['-u', f'secondary_filenames="{cjoin(secondary)}"'])

        self.filename = (Path(self.docdir) / (self.tag + '.pdf')).resolve()
        asycmd.extend(['-o', self.filename])

        return [str(x) for x in asycmd]

    def make(self):
        asycommand = self.asycmd()
        logging.info('ASY: ' + sjoin(asycommand))

        fout = tempfile.TemporaryFile(mode="w+")
        ferr = tempfile.TemporaryFile(mode="w+")
        
        
        proc = subprocess.Popen(asycommand, cwd=top, stdout=fout, stderr=ferr)
        
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logging.info("Asy command killed: " + sjoin(asycommand))
            proc.kill()
            
            fout.seek(0)
            ferr.seek(0)
            cout = fout.read()
            cerr = ferr.read()

            print(cout)
            print(cerr)

        
        if proc.returncode != 0:
            logging.warn('ASY command failed: ' + sjoin(asycommand))

            fout.seek(0)
            ferr.seek(0)
            cout = fout.read()
            cerr = ferr.read()

            print(cout)
            print(cerr)
            


gflopstext = '''\
GFLOP/s are computed based on the Cooley--Tukey operation count \
for a radix-2 transform, and half that for in the case of \
real-complex transforms.  The rocFFT operation count may differ from \
this value: GFLOP/s is provided for the sake of comparison only.'''


efficiencytext = '''\
Efficiency is computed for an idealised FFT which requires exactly \
one read and one write to global memory.  In practice, this \
isn't possible for most problem sizes, as the data does \
not fit into cache, and one must use global memory to store \
intermediary results.  As FFTs are bandwidth-limited on modern hardware, \
the efficiency is measured against the theoretical maximum bandwidth \
for the device.'''


def make_tex(figs, docdir, outdirs, label, secondtype=None):
    """Generate PDF containing performance figures."""

    
    docdir = Path(docdir)

    header = '''\
\\documentclass[12pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{graphicx}
\\usepackage{url}
\\usepackage{hyperref}
\\usepackage{float}
\\begin{document}
\\hypersetup{
  pdfborder={0,0,0},
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue
}
'''
    tex = header

    tex += "\n\\section{Introduction}\n"

    # tex += "Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95\\% confidence interval for the median.  All transforms are " + precision + "-precision.\n\n"

    if secondtype == "gflops":
        tex += gflopstext + "\n\n"

    tex += "\\vspace{1cm}\n"

    tex += "\n\\section{Device Specification}\n"
    for idx in range(len(outdirs)):
        tex += "\n\\subsection{" + str(label[idx])  + "}\n"
        path = Path(outdirs[idx]) / "specs.txt"
        if path.is_file():
            specs = path.read_text()

            for line in specs.split("\n"):
                if line.startswith("Host info"):
                    tex += "\\noindent " + line
                    tex += "\\begin{itemize}\n"
                elif line.startswith("Device info"):
                    tex += "\\end{itemize}\n"
                    tex += line
                    tex += "\\begin{itemize}\n"
                else:
                    if line.strip() != "":
                        tex += "\\item \\verb|" + line + "|\n"
            tex += "\\end{itemize}\n"
            tex += "\n"

    tex += "\\clearpage\n"
    tex += "\\section{Figures}\n"
    tex += "\\listoffigures\n"
    tex += "\\clearpage\n"

    ncompare = 0
    nspeedup = 0
    nslowdown = 0


    figtex = ""
    
    for idx, fig in enumerate(figs):
        figtex += '''
\\centering
\\begin{figure}[H]
   \\includegraphics[width=\\textwidth]{'''
        figtex += str(fig.filename.name)
        figtex += '''}
   \\caption{''' + fig.caption + '''}
\\end{figure}
'''
        for p in fig.secondary:
            
            df = pandas.read_csv(p, sep="\t", comment='#')

            ncompare += len(df.index)
            
            # Significant results:
            df_sig = df.loc[df['speedup_pval'] < 0.05]

            # Significant results that are good or bad:
            df_good = df_sig.loc[df_sig['speedup'] > 1]
            df_bad = df_sig.loc[df_sig['speedup'] < 1]

            if not df_good.empty:
                nspeedup += len(df_good.index)
                
                figtex += "\\begin{table}[H]\n"
                figtex += "\\centering\n"
                figtex += "\\begin{tabular}{l|l|l|}\n"
                figtex += "transform & speedup \% & significance\\\\ \n"
                figtex += "\\hline\n"
                for row in df_good.itertuples(index=False):
                    #figtex += str(row.token).replace("_", "\\_")
                    #figtex += "token"
                    transform_type, placeness, length, batch, precision = perflib.utils.parse_token(row.token)
                    figtex += "$" + "\\times{}".join(str(x) for x in length) + "$"

                    
                    speedup = '{0:.3f}'.format((row.speedup - 1) * 100)
                    pval = '{0:.3f}'.format(row.speedup_pval)
                    figtex += " & " + str(speedup)+ " & " + str(pval) +  "\\\\"
                figtex += "\\hline\n"
                figtex += "\\end{tabular}\n"
                figtex += "\\caption{Improvements for " + fig.caption + "}\n"
                figtex += "\\end{table}\n"
            
            if not df_bad.empty:
                nslowdown += len(df_bad.index)
                
                figtex += "\\begin{table}[H]\n"
                figtex += "\\centering\n"
                figtex += "\\begin{tabular}{l|l|l|}\n"
                figtex += "transform & slowdown \% & significance\\\\ \n"
                figtex += "\\hline\n"
                for row in df_bad.itertuples(index=False):
                    #figtex += str(row.token).replace("_", "\\_")
                    #figtex += "token"
                    transform_type, placeness, length, batch, precision = perflib.utils.parse_token(row.token)
                    figtex += "$" + "\\times{}".join(str(x) for x in length) + "$"

                    if np.prod(batch) > 1:
                        figtex += " by $" + "\\times{}".join(str(x) for x in batch) + "$"
                    
                    speedup = '{0:.3f}'.format((1 - row.speedup) * 100)
                    pval = '{0:.3f}'.format(row.speedup_pval)
                    figtex += " & " + str(speedup)+ " & " + str(pval) +  "\\\\"
                figtex += "\\hline\n"
                figtex += "\\end{tabular}\n"
                figtex += "\\caption{Regressions for " + fig.caption + "}\n"
                figtex += "\\end{table}\n"
            
        figtex += "\\clearpage\n"

    print("ncompare:", ncompare)
    print("nspeedup:", nspeedup)
    print("nslowdown:", nslowdown)
    
    tex += "\\begin{table}[H]\n"
    tex += "\\centering\n"
    tex += "\\begin{tabular}{l|l|l|}\n"
    tex += "ncompare & nspeedup & nslowdown\\\\ \n"
    tex += "\\hline\n"
    tex += str(ncompare) + "&" + str(nspeedup) + "&" + str(nslowdown) + "\\\\\n"
    tex += "100\\%" + "&" + '{0:.3f}'.format(100 * nspeedup / ncompare) + "\\% "+ "&" +  '{0:.3f}'.format(100 * nslowdown / ncompare) + "\\% "+ "\\\\\n"
    tex += "\\hline\n"
    tex += "\\end{tabular}\n"
    tex += "\\caption{Overall Performance Changes}\n"
    tex += "\\end{table}\n"
    tex += "\\clearpage\n"

    tex += figtex
    
    tex += "\n\\end{document}\n"

    fname = docdir / 'figs.tex'
    fname.write_text(tex)

    log = docdir / 'tex.log'
    with log.open('w') as f:
        latexcmd = ['latexmk', '-pdf', fname.name]
        texproc = subprocess.Popen(latexcmd, cwd=fname.parent, stdout=f, stderr=f)
        try:
            texproc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logging.info("tex command killed: " + sjoin(latexcmd))
            texproc.kill()
            
            
        if texproc.returncode != 0:
            print("****tex fail****")
