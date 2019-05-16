#!/usr/bin/env python
# #############################################################################
# Copyright (c) 2013 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

import datetime
import sys
import argparse
import subprocess
import itertools
import os
import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('agg')
import pylab
from matplotlib.backends.backend_pdf import PdfPages
from fftPerformanceTesting import *

def plotGraph(dataForAllPlots, title, plottype, plotkwargs, xaxislabel, yaxislabel):
  """
  display a pretty graph
  """
  dh.write('Making graph\n')
  colors = ['k','y','m','c','b','r','g']
  #plottype = 'plot'
  for thisPlot in dataForAllPlots:
    getattr(pylab, plottype)(thisPlot.xdata, thisPlot.ydata,
                             '{}.-'.format(colors.pop()),
                             label=thisPlot.label, **plotkwargs)
  if len(dataForAllPlots) > 1:
    pylab.legend(loc='best')

  pylab.title(title)
  pylab.xlabel(xaxislabel)
  pylab.ylabel(yaxislabel)
  pylab.grid(True)

  if args.outputFilename == None:
    # if no pdf output is requested, spit the graph to the screen . . .
    pylab.show()
  else:
    pylab.savefig(args.outputFilename,dpi=(1024/8))
    # . . . otherwise, gimme gimme pdf
    #pdf = PdfPages(args.outputFilename)
    #pdf.savefig()
    #pdf.close()

######## plotFromDataFile() Function to plot from data file begins ########
def plotFromDataFile():
  data = []
  """
  read in table(s) from file(s)
  """
  for thisFile in args.datafile:
    if not os.path.isfile(thisFile):
      print 'No file with the name \'{}\' exists. Please indicate another filename.'.format(thisFile)
      quit()

    results = open(thisFile, 'r')
    resultsContents = results.read()
    resultsContents = resultsContents.rstrip().split('\n')

    raw_data = []
    for line in resultsContents:
        if not (line.startswith('#') or len(line.strip()) == 0):
            raw_data.append(line.split('#')[0].rstrip(', '))

    #firstRow = raw_data.pop(0)
    #if firstRow != tableHeader:
    #  print 'ERROR: input file \'{}\' does not match expected format.'.format(thisFile)
    #  quit()

    for row in raw_data:
      row = row.split(',')
      row = TableRow(TestCombination(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]), row[10])
      data.append(GraphPoint(row.parameters.x, row.parameters.y, row.parameters.z, row.parameters.batchsize, row.parameters.precision, row.parameters.device, row.parameters.label, row.gflops))

  """
  data sanity check
  """
  # if multiple plotvalues have > 1 value among the data rows, the user must specify which to plot
  multiplePlotValues = []
  for option in plotvalues:
    values = []
    for point in data:
      values.append(getattr(point, option))
    multiplePlotValues.append(len(set(values)) > 1)
  if multiplePlotValues.count(True) > 1 and args.plot == None:
    print 'ERROR: more than one parameter of {} has multiple values. Please specify which parameter to plot with --plot'.format(plotvalues)
    quit()

  # if args.graphxaxis is not 'problemsize', the user should know that the results might be strange
  if args.graphxaxis != 'problemsize':
    xaxisvalueSet = []
    for option in xaxisvalues:
      if option != 'problemsize':
        values = []
        for point in data:
          values.append(getattr(point, option))
        xaxisvalueSet.append(len(set(values)) > 1)
    if xaxisvalueSet.count(True) > 1:
      print 'WARNING: more than one parameter of {} is varied. unexpected results may occur. please double check your graphs for accuracy.'.format(xaxisvalues)

  # multiple rows should not have the same input values
  pointInputs = []
  for point in data:
    pointInputs.append(point.__str__().split(';')[0])
  if len(set(pointInputs)) != len(data):
    print 'ERROR: imported table has duplicate rows with identical input parameters'
    quit()

  """
  figure out if we have multiple plots on this graph (and what they should be)
  """
  if args.plot != None:
    multiplePlots = args.plot
  elif multiplePlotValues.count(True) == 1:
    multiplePlots = plotvalues[multiplePlotValues.index(True)]
  else:
    # default to device if none of the options to plot have multiple values
    multiplePlots = 'device'

  """
  assemble data for the graphs
  """
  data.sort(key=lambda row: int(getattr(row, args.graphxaxis)))

  # choose scale for x axis
  if args.xaxisscale == None:
    # user didn't specify. autodetect
    if int(getattr(data[len(data)-1], args.graphxaxis)) > 2000: # big numbers on x-axis
      args.xaxisscale = 'log2'
    elif int(getattr(data[len(data)-1], args.graphxaxis)) > 10000: # bigger numbers on x-axis
      args.xaxisscale = 'log10'
    else: # small numbers on x-axis
      args.xaxisscale = 'linear'

  if args.yaxisscale == None:
    args.yaxisscale = 'linear'

  plotkwargs = {}
  if args.xaxisscale == 'linear':
    plottype = 'plot'
  elif args.xaxisscale == 'log2':
    plottype = 'semilogx'
    if (args.yaxisscale=='log2'):
      plottype = 'loglog'
      plotkwargs = {'basex':2,'basey':2}
    elif (args.yaxisscale=='log10'):
      plottype = 'loglog'
      plotkwargs = {'basex':2,'basey':10}
    elif (args.yaxisscale=='linear'):
      plottype = 'semilogx'
      plotkwargs = {'basex':2}
  elif args.xaxisscale == 'log10':
    plottype = 'semilogx'
    if (args.yaxisscale=='log2'):
      plottype = 'loglog'
      plotkwargs = {'basex':10,'basey':2}
    elif (args.yaxisscale=='log10'):
      plottype = 'loglog'
      plotkwargs = {'basex':10,'basey':10}
  else:
    print 'ERROR: invalid value for x-axis scale'
    quit()


  plots = set(getattr(row, multiplePlots) for row in data)

  class DataForOnePlot:
    def __init__(self, inlabel, inxdata, inydata):
      self.label = inlabel
      self.xdata = inxdata
      self.ydata = inydata

  dataForAllPlots=[]
  for plot in plots:
    dataForThisPlot = itertools.ifilter( lambda x: getattr(x, multiplePlots) == plot, data)
    dataForThisPlot = list(itertools.islice(dataForThisPlot, None))
    if args.graphxaxis == 'problemsize':
      xdata = [int(row.x) * int(row.y) * int(row.z) * int(row.batchsize) for row in dataForThisPlot]
    else:
      xdata = [getattr(row, args.graphxaxis) for row in dataForThisPlot]
    ydata = [getattr(row, args.graphyaxis) for row in dataForThisPlot]
    dataForAllPlots.append(DataForOnePlot(plot,xdata,ydata))

  """
  assemble labels for the graph or use the user-specified ones
  """
  if args.graphtitle:
    # use the user selection
    title = args.graphtitle
  else:
    # autogen a lovely title
    title = 'Performance vs. ' + args.graphxaxis.capitalize()

  if args.xaxislabel:
    # use the user selection
    xaxislabel = args.xaxislabel
  else:
    # autogen a lovely x-axis label
    if args.graphxaxis == 'cachesize':
      units = '(bytes)'
    else:
      units = '(datapoints)'

    xaxislabel = args.graphxaxis + ' ' + units

  if args.yaxislabel:
    # use the user selection
    yaxislabel = args.yaxislabel
  else:
    # autogen a lovely y-axis label
    if args.graphyaxis == 'gflops':
      units = 'GFLOPS'
    yaxislabel = 'Performance (' + units + ')'

  """
  display a pretty graph
  """
  colors = ['k','y','m','c','b','g','r']

  def getkey(item):
    return str(item.label)
  dataForAllPlots.sort(key=getkey)

  if len(dataForAllPlots) > 18: #todo, better color scheme
    colors = list(matplotlib.colors.cnames.values())
  elif len(dataForAllPlots) > 7:
    # follow https://xkcd.com/color/rgb/
    colors = [u'#e50000', u'#15b01a', u'#0343df', u'#ff81c0', u'#653700', u'#7e1e9c', \
              u'#ffff14', u'#029386', u'#f97306', u'#96f97b', u'#c20078', u'#95d0fc', \
              u'#75bbfd', u'#929591', u'#89fe05', u'#bf77f6', u'#9a0eea', u'#033500'  ]

  for thisPlot in sorted(dataForAllPlots,key=getkey):
    getattr(pylab, plottype)(thisPlot.xdata, thisPlot.ydata, colors.pop(), label=thisPlot.label, **plotkwargs)

  if len(dataForAllPlots) > 1:
    pylab.legend(loc='best')

  pylab.title(title)
  pylab.xlabel(xaxislabel)
  pylab.ylabel(yaxislabel)
  pylab.grid(True)

  if args.outputFilename == None:
    # if no pdf output is requested, spit the graph to the screen . . .
    pylab.show()
  else:
    # . . . otherwise, gimme gimme pdf
    #pdf = PdfPages(args.outputFilename)
    #pdf.savefig()
    #pdf.close()
    pylab.savefig(args.outputFilename,dpi=(1024/8))
######### plotFromDataFile() Function to plot from data file ends #########



######## "main" program begins #####
"""
define and parse parameters
"""

xaxisvalues = ['x','y','z','batchsize','problemsize']
yaxisvalues = ['gflops']
plotvalues = ['device', 'precision', 'label']



parser = argparse.ArgumentParser(description='Plot performance of the clfft\
    library. clfft.plotPerformance.py reads in data tables from clfft.\
    measurePerformance.py and plots their values')
fileOrDb = parser.add_mutually_exclusive_group(required=True)
fileOrDb.add_argument('-d', '--datafile',
  dest='datafile', action='append', default=None, required=False,
  help='indicate a file to use as input. must be in the format output by\
  clfft.measurePerformance.py. may be used multiple times to indicate\
  multiple input files. e.g., -d cypressOutput.txt -d caymanOutput.txt')
parser.add_argument('-x', '--x_axis',
  dest='graphxaxis', default=None, choices=xaxisvalues, required=True,
  help='indicate which value will be represented on the x axis. problemsize\
      is defined as x*y*z*batchsize')
parser.add_argument('-y', '--y_axis',
  dest='graphyaxis', default='gflops', choices=yaxisvalues,
  help='indicate which value will be represented on the y axis')
parser.add_argument('--plot',
  dest='plot', default=None, choices=plotvalues,
  help='indicate which of {} should be used to differentiate multiple plots.\
      this will be chosen automatically if not specified'.format(plotvalues))
parser.add_argument('--title',
  dest='graphtitle', default=None,
  help='the desired title for the graph generated by this execution. if\
      GRAPHTITLE contains any spaces, it must be entered in \"double quotes\".\
      if this option is not specified, the title will be autogenerated')
parser.add_argument('--x_axis_label',
  dest='xaxislabel', default=None,
  help='the desired label for the graph\'s x-axis. if XAXISLABEL contains\
      any spaces, it must be entered in \"double quotes\". if this option\
      is not specified, the x-axis label will be autogenerated')
parser.add_argument('--x_axis_scale',
  dest='xaxisscale', default=None, choices=['linear','log2','log10'],
  help='the desired scale for the graph\'s x-axis. if nothing is specified,\
      it will be selected automatically')
parser.add_argument('--y_axis_scale',
  dest='yaxisscale', default=None, choices=['linear','log2','log10'],
  help='the desired scale for the graph\'s y-axis. if nothing is specified,\
      linear will be selected')
parser.add_argument('--y_axis_label',
  dest='yaxislabel', default=None,
  help='the desired label for the graph\'s y-axis. if YAXISLABEL contains any\
      spaces, it must be entered in \"double quotes\". if this option is not\
      specified, the y-axis label will be autogenerated')
parser.add_argument('--outputfile',
  dest='outputFilename', default=None,
  help='name of the file to output graphs. Supported formats: emf, eps, pdf, png, ps, raw, rgba, svg, svgz.')

args = parser.parse_args()

if args.datafile != None:
  plotFromDataFile()
else:
  print "Atleast specify if you want to use text files or database for plotting graphs. Use -h or --help option for more details"
  quit()

