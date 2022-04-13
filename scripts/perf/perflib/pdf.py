"""Utilities to generate PDF plots (via Asymptote)."""

import logging
import subprocess

from dataclasses import dataclass
from pathlib import Path
from perflib.utils import sjoin, cjoin
from typing import List
import tempfile

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

        if self.figtype == "linegraph":
            asycmd.append(top / "datagraphs.asy")
        elif self.figtype == "bargraph":
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


def make_tex(figs, docdir, outdirs, secondtype=None):
    """Generate PDF containing performance figures."""

    docdir = Path(docdir)

    header = '''\
\\documentclass[12pt]{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{graphicx}
\\usepackage{url}
\\usepackage{hyperref}
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
        tex += "\n\\subsection{" + str(outdirs[idx])  + "}\n"
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

    for idx, fig in enumerate(figs):
        tex += '''
\\centering
\\begin{figure}[h]
   \\includegraphics[width=\\textwidth]{'''
        tex += str(fig.filename.name)
        tex += '''}
   \\caption{''' + fig.caption + '''}
\\end{figure}
'''
        tex += "\\clearpage\n"

    tex += "\n\\end{document}\n"

    fname = docdir / 'figs.tex'
    fname.write_text(tex)

    log = docdir / 'tex.log'
    with log.open('w') as f:
        latexcmd = ['latexmk', '-pdf', fname.name]
        texproc = subprocess.run(latexcmd, cwd=fname.parent, stdout=f, stderr=f)
        if texproc.returncode != 0:
            print("****tex fail****")
