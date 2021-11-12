import subprocess
import pathlib

import docx

import perflib.docx_emf_patch


def pdf2emf(path: pathlib.Path):
    """Convert PDF to EMF."""
    pdf, svg, emf = str(path), str(path.with_suffix(".svg")), str(path.with_suffix(".emf"))
    subprocess.check_call(["pdf2svg", pdf, svg])
    # Older versions of inkscape use -M.
    #subprocess.check_call(["inkscape", svg, "-M", emf])
    subprocess.check_call(["inkscape", svg, "--export-filename", emf])
    return emf


def make_docx(figs, docdir, outdirs, secondtype=None):

    docdir = pathlib.Path(docdir)

    document = docx.Document()

    document.add_heading('rocFFT benchmarks', 0)

    # document.add_paragraph("Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95% confidence interval for the median.  Transforms are " + precision + "-precision, forward, and in-place.")

    # if secondtype == "gflops":
    #     document.add_paragraph(gflopstext)
    # if secondtype == "efficiency":
    #     document.add_paragraph(efficiencytext)

    # specfilename = os.path.join(outdir, "specs.txt")
    # if os.path.isfile(specfilename):
    #     with open(specfilename, "r") as f:
    #         specs = f.read()
    #     for line in specs.split("\n"):
    #         document.add_paragraph(line)

    for fig in figs:
        emfname = pdf2emf(fig.filename)
        document.add_picture(emfname, width=docx.shared.Inches(6))
        document.add_paragraph(fig.caption)

    document.save(str(docdir / 'figs.docx'))
