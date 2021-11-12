"""HTML plotting."""

import os

from .pdf import BaseFigure
from .utils import to_data_frames

from plotly import graph_objs as go


def speedup_saturation(speedup):
    diff = abs(1.0 - speedup)
    # scale up a 20% difference into maximum 50% saturation - lower
    # difference should be closer to 255 (white)
    saturation_ratio = min(0.5, diff / 0.2 * 0.5)
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


def title_to_html_anchor(title):
    return title.replace(' ', '_')


class HTMLFigure(BaseFigure):

    def make(self):
        data_frames = to_data_frames(self.primary, self.secondary)
        logscale = False

        traces = []

        traces.append(go.Scatter(
            x=data_frames[0].elements,
            y=data_frames[0].median_sample,
            hovertext=data_frames[0].length,
            name=self.labels[0]
        ))

        for i in range(1,len(data_frames)):
            traces.append(go.Scatter(
                x=data_frames[i].elements,
                y=data_frames[i].median_sample,
                hovertext=data_frames[i].length,
                name=self.labels[i]
            ))
            traces.append(go.Scatter(
                x=data_frames[i].elements,
                y=data_frames[i].speedup,
                name='Speedup {} over {}'.format(self.labels[i], self.labels[0]),
                yaxis='y2',
                error_y = dict(
                    type='data',
                    symmetric=False,
                    array=data_frames[i].speedup_high - data_frames[i].speedup,
                    arrayminus=data_frames[i].speedup - data_frames[i].speedup_low,
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
            title=self.title,
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
            x0=data_frames[0].elements.min(),
            y0=1,
            x1=data_frames[0].elements.max(),
            y1=1,
            line=dict(color='grey', dash='dash'),
            yref='y2'
        )

        nrows = len(data_frames[0].index)
        headers = ['Problem size', 'Elements']
        values = [
            data_frames[0].length,
            data_frames[0].elements
        ]
        fill_colors = [
            ['white'] * nrows,
            ['white'] * nrows,
        ]
        for i in range(len(data_frames)):
            headers.append(self.labels[i] + ' (median)')
            values.append(["{:.4f}".format(x) for x in data_frames[i].median_sample])
            fill_colors.append(['white'] * nrows)

            if i > 0:
                headers.append('Speedup {} over {}'.format(self.labels[i], self.labels[0]))
                values.append(["{:.4f}".format(x) for x in data_frames[i].speedup])
                fill_colors.append(speedup_colors(data_frames[i].speedup))

                headers.append('Significance of {} over {}'.format(self.labels[i], self.labels[0]))
                values.append(["{:.4f}".format(x) for x in data_frames[i].speedup_pval])
                fill_colors.append(significance_colors(data_frames[i].speedup_pval))

        table = go.Figure(
            data = [
                go.Table(header=dict(values=headers),
                         cells=dict(values=values, fill_color=fill_colors),
                         ),
            ],
            layout = go.Layout(
                title=self.title,
            )
        )
        # 900 seems to be enough for 2 dirs
        # widen table by 200px per extra dir
        table_width = 900
        if len(data_frames) > 2:
            table_width += (len(data_frames) - 2) * 200
        table.update_layout(width=table_width, height=600)

        self.plot = fig
        self.table = table


def make_html(figures, title, docdir, outdirs):
    # TODO: this needs to read the output from the post-processing;
    # graphing and post-processing should be separate.

    # # use first dir's dat files as a basis for what to graph.
    # # assumption is that other dirs have the same-named files.
    # dat_files = glob.glob(os.path.join(dirs[0], '*.dat'))
    # # sort files so diagrams show up in consistent order for each run
    # dat_files.sort()

    # construct the output file "figs.html"
    outfile = open(docdir / 'figs.html', 'w')
    outfile.write('''
<html>
  <head>
    <title>{}</title>
  </head>
  <body>
'''.format(title))

    # make links to each figure at the top of the report
    outfile.write('''<b>Quick links:</b><br/>''')
    for figure in figures:
        plot, table = figure.plot, figure.table
        fig_title = plot.layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<a href="#{}">{}</a><br/>'''.format(anchor, fig_title))

    # include specs in the report
    outfile.write('''<table><tr>''')
    for d in outdirs:
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
    for figure in figures:
        plot, table = figure.plot, figure.table
        fig_title = plot.layout.title.text
        anchor = title_to_html_anchor(fig_title)
        outfile.write('''<tr><td id="{}">'''.format(anchor))

        outfile.write(plot.to_html(full_html=False, include_plotlyjs=include_js))
        include_js = False
        outfile.write('''</td><td>''')
        outfile.write(table.to_html(full_html=False, include_plotlyjs=False))
        outfile.write('''</td></tr>''')

    outfile.write('''
    </table>
    </body>
    </html>
    ''')
