#!/usr/bin/env python3
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import math
import sys
import functools
import glob
import os
import scipy.stats
import shutil
import random

from pathlib import Path

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

    for cur_lengths_str, cur_sample in data_dict.items():
        if not cur_sample.data:
            continue

        cur_lengths = [int(x) for x in cur_lengths_str.split('x')]

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
def basename_to_title(basename):
    title = ''
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

def get_title(path):
    """Read title from '# title: ' line in `basename`, or generate from `basename`."""

    try:
        lines = path.read_text().splitlines()
    except:
        lines = []

    for line in lines:
        if line.startswith('# title: '):
            return line[9:].strip()

    return basename_to_title(path.name)


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

def speedup_saturation(speedup):
    diff = abs(1.0 - speedup)
    # scale up a 20% difference into maximum 50% saturation - lower
    # difference should be closer to 255 (white)
    saturation_ratio = min(0.5,diff / 0.2 * 0.5)
    return 255 - int(saturation_ratio * 255.0)

def speedup_colors(speedup):
    ret = []
    for s in speedup:
        saturation = speedup_saturation(s)
        if s < 1.0:
            # slowdown is red
            ret.append('rgb(255,{0},{0})'.format(saturation))
        else:
            # speedup is green
            ret.append('rgb({0},255,{0})'.format(saturation))
    return ret

def significance_colors(significance, threshold=0.05):
    ret = []
    for s in significance:
        if s < threshold:
            saturation = 128 + int(128 * (s / threshold))
            ret.append('rgb({0},{0},255)'.format(saturation))
        else:
            ret.append('white')
    return ret

# returns a tuple of the plotly graph object and the table object with
# the same data
def graph_file(basename, dirs, logscale, docdir):
    dir_basenames = []
    dir_dirnames = []
    data_dicts = []
    data_frames = []
    traces = []
    for dir in dirs:
        dir = os.path.normpath(dir)
        try:
            dd = file_to_data_dict(os.path.join(dir,basename))
            for samples in dd.values():
                if len(samples.data) < 1:
                    raise ValueError
        except:
            continue

        dir_basenames.append(os.path.basename(dir))
        dir_dirnames.append(os.path.dirname(dir))
        data_dicts.append(dd)
        data_frames.append(data_dict_to_frame(dd))

    if not data_frames:
        return

    for i in range(len(data_frames)):
        data_frames[i].set_index('lengths')
        data_frames[i].sort_values('num_elems', inplace=True)
        if i > 0:
            data_frames[i].reindex(index=data_frames[0].index)

    for i in range(1, len(data_frames)):
        pvalues = []
        for l in data_frames[0].lengths:
            _, p, _, _ = scipy.stats.median_test(data_dicts[i][l].data, data_dicts[0][l].data)
            pvalues.append(p)
        data_frames[i] = data_frames[i].assign(
            # speedup and speedup confidence interval
            speedup=data_frames[0].median_sample / data_frames[i].median_sample,
            pvalue=pvalues,
            # FIXME: we're doing speedup_confidence twice, which is
            # unnecessary
            speedup_errlow=lambda x: [x[0] for x in speedup_confidence(x.lengths, data_dicts[0], data_dicts[i])],
            speedup_errhigh=lambda x: [x[1] for x in speedup_confidence(x.lengths, data_dicts[0], data_dicts[i])],
        )

    # initial line
    traces.append(go.Scatter(
        x=data_frames[0].num_elems,
        y=data_frames[0].median_sample,
        hovertext=make_hovertext(data_frames[0].lengths, data_frames[0].batches),
        name=dir_basenames[0]
    ))

    for i in range(1,len(data_frames)):
        traces.append(go.Scatter(
            x=data_frames[i].num_elems,
            y=data_frames[i].median_sample,
            hovertext=make_hovertext(data_frames[i].lengths, data_frames[i].batches),
            name=dir_basenames[i]
        ))
        traces.append(go.Scatter(
            x=data_frames[i].num_elems,
            y=data_frames[i].speedup,
            name='Speedup {} over {}'.format(dir_basenames[i], dir_basenames[0]),
            yaxis='y2',
            error_y = dict(
                type='data',
                symmetric=False,
                array=data_frames[i].speedup_errhigh,
                arrayminus=data_frames[i].speedup_errlow,
            )
        ))
        
    if logscale:
        x_title = 'Problem size (elements, logarithmic)'
        axis_type = 'log'
        y_title = 'Time (ms, logarithmic)'
    else:
        x_title = 'Problem size (elements)'
        axis_type = 'linear'
        y_title = 'Time (ms)'

    layout = go.Layout(
        title=get_title(Path(dirs[0]) / basename),
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

    fig = go.Figure(data=traces, layout=layout)
    # add speedup=1 reference line
    fig.add_shape(
        type='line',
        x0=data_frames[0].num_elems.min(),
        y0=1,
        x1=data_frames[0].num_elems.max(),
        y1=1,
        line=dict(color='grey', dash='dash'),
        yref='y2'
    )

    nrows = len(data_frames[0].index)
    headers = ['Problem size', 'Elements']
    values = [
        [f"{length} b{batch}" for length, batch in zip(data_frames[0].lengths, data_frames[0].batches)],
        data_frames[0].num_elems
    ]
    fill_colors = [
        ['white'] * nrows,
        ['white'] * nrows,
    ]
    for i in range(len(data_frames)):
        headers.append(dir_basenames[i] + ' (median)')
        values.append(["{:.4f}".format(x) for x in data_frames[i].median_sample])
        fill_colors.append(['white'] * nrows)

        if i > 0:
            headers.append('Speedup {} over {}'.format(dir_basenames[i], dir_basenames[0]))
            values.append(["{:.4f}".format(x) for x in data_frames[i].speedup])
            fill_colors.append(speedup_colors(data_frames[i].speedup))

            headers.append('Significance of {} over {}'.format(dir_basenames[i], dir_basenames[0]))
            values.append(["{:.4f}".format(x) for x in data_frames[i].pvalue])
            fill_colors.append(significance_colors(data_frames[i].pvalue))

    table = go.Figure(
        data = [
            go.Table(header=dict(values=headers),
                     cells=dict(values=values, fill_color=fill_colors),
                     ),
        ],
        layout = go.Layout(
            title=get_title(Path(dirs[0]) / basename)
            )
    )
    # 900 seems to be enough for 2 dirs
    # widen table by 200px per extra dir
    table_width = 900
    if len(dirs) > 2:
        table_width += (len(dirs) - 2) * 200
    table.update_layout(width=table_width, height=600)

    return (fig,table)

def title_to_html_anchor(title):
    return title.replace(' ', '_')

def graph_dirs(dirs, title, docdir):
    # use first dir's dat files as a basis for what to graph.
    # assumption is that other dirs have the same-named files.
    dat_files = glob.glob(os.path.join(dirs[0], '*.dat'))
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

    # collect a list of figure+table pairs
    figs_list = []
    for filename in dat_files:
        g = graph_file(os.path.basename(filename), dirs, True, docdir)
        if g is not None:
            figs_list.append(g)
        else:
            print(f'Skipped {filename}...')

    # make links to each figure at the top of the report
    outfile.write('''<b>Quick links:</b><br/>''')
    for figs in figs_list:
        fig_title = figs[0].layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<a href="#{}">{}</a><br/>'''.format(anchor, fig_title))

    # include specs in the report
    outfile.write('''<table><tr>''')
    for d in dirs:
        outfile.write('''<td><b>{} specs:</b><br/><pre>'''.format(
            os.path.basename(os.path.normpath(d)))
        )
        specs_path = os.path.join(d, 'specs.txt')
        try:
            with open(specs_path) as specs:
              for line in specs:
                  outfile.write(line)
        except FileNotFoundError:
            outfile.write("N/A")

        outfile.write('''</pre></td>''')
    outfile.write('''</tr></table><br/><table>''')

    # only the first figure needs js included
    include_js = True
    for figs in figs_list:
        fig_title = figs[0].layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<tr><td id="{}">'''.format(anchor))

        outfile.write(figs[0].to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False
        outfile.write('''</td><td>''')
        outfile.write(figs[1].to_html(full_html=False, include_plotlyjs=False))
        outfile.write('''</td></tr>''')

    outfile.write('''
    </table>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    graph_dirs(sys.argv[1:-1], 'Performance report', sys.argv[-1])
