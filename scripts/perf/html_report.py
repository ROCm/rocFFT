#!/usr/bin/env python3
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import sys
import functools
import glob
import os
import shutil
import random

class Sample:
    def __init__(self, data, batch):
        self.data = data
        self.batch = batch

# returns a dict that maps a length string (e.g. 16x27) to Samples
def file_to_data_dict(filename):
    infile = open(filename, 'r')

    data_dict = {}

    for line in infile:
        # ignore comments
        if line.startswith('#'):
            continue
        words = line.split("\t")
        dim = int(words[0])
        cur_lengths = 'x'.join(words[1:dim+1])
        cur_batch = int(words[dim+1])
        cur_samples = list(map(lambda x : float(x), words[dim+3:]))
        data_dict[cur_lengths] = Sample(cur_samples, cur_batch)
    return data_dict

# convert raw data dict to data frame suitable for graphing
def data_dict_to_frame(data_dict):
    num_elems = []
    lengths = []
    samples = []
    batches = []
    min_samples = sys.maxsize

    for cur_lengths_str, cur_sample in data_dict.items():
        cur_lengths = [int(x) for x in cur_lengths_str.split('x')]
        if len(cur_sample.data) < min_samples:
            min_samples = len(cur_sample.data)

        num_elems.append(functools.reduce((lambda x, y: x * y), cur_lengths))
        lengths.append(cur_lengths_str)
        samples.append(cur_sample.data)
        batches.append(cur_sample.batch)

    median_samples = []
    max_samples = []
    min_samples = []
    for s in samples:
        s.sort()
        median_samples.append(s[len(s)//2])
        max_samples.append(s[-1])
        min_samples.append(s[0])

    data = pd.DataFrame(
        {
        'num_elems': num_elems,
        'lengths': lengths,
        'median_sample': median_samples,
        'max_sample': max_samples,
        'min_sample': min_samples,
        'batches': batches,
        }
        )

    return data

# decode the filename into a nicer human-readable string
def filename_to_title(filename):
    title = ''
    basename = os.path.basename(filename)
    # dimension
    if 'dim1' in basename:
        title += '1D '
    elif 'dim2' in basename:
        title += '2D '
    elif 'dim3' in basename:
        title += '3D '

    # precision
    if 'double' in basename:
        title += 'double-precision '
    elif 'single' in basename:
        title += 'single-precision '

    # direction
    if '_inv_' in basename:
        title += 'inverse '
    else:
        title += 'forward '

    # transform type
    if 'c2c' in basename:
        title += 'C2C '
    elif 'r2c' in basename:
        title += 'R2C '

    # placement
    if 'inplace' in basename:
        title += 'in-place'
    elif 'outofplace' in basename:
        title += 'out-of-place'
    return title

# return tuple with low,high to define the interval
def speedup_confidence(length_series, dir0_data_dict, dir1_data_dict):
    ret = []
    for length in length_series:
        # do a bunch of random samples of speedup between dir0 and dir1 for this length
        samples = []
        for _ in range(50):
            dir0_choice = random.choice(dir0_data_dict[length].data)
            dir1_choice = random.choice(dir1_data_dict[length].data)
            # compute speedup between those choices
            samples.append(dir0_choice / dir1_choice)

        # work out confidence interval for those random samples
        samples_mean = np.mean(samples)
        std = np.std(samples)
        n = len(samples)
        # 95% CI
        z = 1.96
        lower = samples_mean - (z * (std/math.sqrt(n)))
        upper = samples_mean + (z * (std/math.sqrt(n)))
        # NOTE: plotly wants an absolute difference from the mean.
        # normal confidence interval would be (lower, upper)
        ret.append((samples_mean - lower,upper - samples_mean))
    return ret

def make_hovertext(lengths, batches):
    return ["{} batch {}".format(length,batch) for length, batch in zip(lengths,batches)]

# returns the plotly figure object
def graph_file(filename, dir1, logscale, docdir):
    dir0 = os.path.dirname(filename)
    dir0_base = os.path.basename(dir0)
    dir1_base = os.path.basename(dir1)

    dir0_data_dict = file_to_data_dict(filename)
    dir1_data_dict = file_to_data_dict(os.path.join(dir1, os.path.basename(filename)))
    dir0_data = data_dict_to_frame(dir0_data_dict)
    dir1_data = data_dict_to_frame(dir1_data_dict)

    graph_data = dir0_data
    graph_data.insert(5, 'median_sample_1', dir1_data['median_sample'])
    graph_data.insert(6, 'max_sample_1', dir1_data['max_sample'])
    graph_data.insert(7, 'min_sample_1', dir1_data['min_sample'])

    graph_data = graph_data.assign(
        # speedup and speedup confidence interval
        speedup=lambda x: x.median_sample / x.median_sample_1,
        # FIXME: we're doing speedup_confidence twice, which is
        # unnecessary
        speedup_errlow=lambda x: [x[0] for x in speedup_confidence(x.lengths, dir0_data_dict, dir1_data_dict)],
        speedup_errhigh=lambda x: [x[1] for x in speedup_confidence(x.lengths, dir0_data_dict, dir1_data_dict)],
    )
    graph_data.sort_values('num_elems',inplace=True)

    # lines for dir0, dir1, speedup
    dir0_trace = go.Scatter(
        x=graph_data['num_elems'],
        y=graph_data['median_sample'],
        hovertext=make_hovertext(graph_data['lengths'],graph_data['batches']),
        name=dir0_base
    )
    dir1_trace = go.Scatter(
        x=graph_data['num_elems'],
        y=graph_data['median_sample_1'],
        hovertext=make_hovertext(graph_data['lengths'],graph_data['batches']),
        name=dir1_base
    )
    speedup_trace = go.Scatter(
        x=graph_data['num_elems'],
        y=graph_data['speedup'],
        name='Speedup',
        yaxis='y2',
        error_y = dict(
            type='data',
            symmetric=False,
            array=graph_data['speedup_errhigh'],
            arrayminus=graph_data['speedup_errlow'],
        )
    )
    if logscale:
        x_title = 'Problem size (elements, logarithmic)'
        axis_type = 'log'
        y_title = 'Time (ms, logarithmic)'
    else:
        x_title = 'Problem size (elements)'
        axis_type = 'linear'
        y_title = 'Time (ms)'
    layout = go.Layout(
        title=filename_to_title(filename),
        xaxis=dict(
            title=x_title,
            type=axis_type,
        ),
        yaxis=dict(
            title=y_title,
            type=axis_type,
            rangemode='tozero'
        ),
        yaxis2=dict(
            title='Speedup',
            overlaying='y',
            side='right',
            type='linear',
            rangemode='tozero'
        ),
        hovermode = 'x unified',
        width = 900,
        height = 600,
        legend = dict(
            yanchor="top",
            xanchor="right",
            x=1.2
        )
    )

    fig = go.Figure(data=[dir0_trace, dir1_trace, speedup_trace], layout=layout)
    # add speedup=1 reference line
    fig.add_shape(
        type='line',
        x0=graph_data['num_elems'].min(),
        y0=1,
        x1=graph_data['num_elems'].max(),
        y1=1,
        line=dict(color='grey', dash='dash'),
        yref='y2'
    )

    return fig

def graph_dirs(dir0, dir1, title, docdir):
    # use dir0's dat files as a basis for what to graph.
    # assumption is that dir1 has the same-named files.
    dat_files = glob.glob(os.path.join(dir0, '*.dat'))
    # sort files so diagrams show up in consistent order for each run
    dat_files.sort()

    # construct the output file "figs.html"
    outfile = open(os.path.join(docdir,'figs.html'), 'w')
    outfile.write('''
<html>
  <head>
    <title>{}</title>
  </head>
  <body>
'''.format(title))

    # only the first figure needs js included
    include_js = True
    for filename in dat_files:
        fig = graph_file(filename, dir1, True, docdir)
        outfile.write(fig.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False

    outfile.write('''
    </body>
    </html>
    ''')

if __name__ == '__main__':
    graph_dirs(sys.argv[1], sys.argv[2], 'Performance report', sys.argv[3])
