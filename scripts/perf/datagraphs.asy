import graph;
import utils;

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

// Input data:
string filenames = "";
string secondary_filenames = "";
string legendlist = "";

// Graph formatting
string xlabel = "Problem size N";
string ylabel = "Time (ms)";

string primaryaxis = "time";
string secondaryaxis = "speedup";

usersetting();

if(primaryaxis == "gflops") {
    ylabel = "GFLOP/s";
}

//write("filenames:\"", filenames+"\"");
if(filenames == "") {
    filenames = getstring("filenames");
}
    
if (legendlist == "") {
    legendlist = filenames;
}

bool myleg = ((legendlist == "") ? false : true);
string[] legends = set_legends(legendlist);
for (int i = 0; i < legends.length; ++i) {
  legends[i] = texify(legends[i]);
}

string[] testlist = listfromcsv(filenames);

// Data containers:
datapoint[][] datapoints = new datapoint[testlist.length][];

readfiles(testlist, datapoints);

pair[][] xyval = new real[testlist.length][];
pair[][] ylowhigh = new real[testlist.length][];
for(int n = 0; n < datapoints.length; ++n) {
    datapoints_to_xyvallowhigh(datapoints[n], xyval[n], ylowhigh[n]);
}

write(xyval);
write(ylowhigh);

// Find the bounds on the data to determine if the scales should be
// logarithmic.
real[] bounds = xyminmax( xyval );
bool xlog = true;
if(bounds[1] / bounds[0] < 10) {
    xlog = false;
}
bool ylog = true;
if(bounds[3] / bounds[2] < 10) {
    ylog = false;
}
scale(xlog ? Log : Linear, ylog ? Log : Linear);

// Plot the primary graph:
for(int n = 0; n < xyval.length; ++n)
{
    pen graphpen = Pen(n);
    if(n == 2) {
        graphpen = darkgreen;
    }
    string legend = myleg ? legends[n] : texify(testlist[n]);
    marker mark = marker(scale(0.5mm) * unitcircle, Draw(graphpen + solid));

    // Compute the error bars:
    pair[] dp; // high
    pair[] dm; // low
    for(int i = 0; i < xyval[n].length; ++i) {
        dp.push((0, -xyval[n][i].y + ylowhigh[n][i].y));
        dm.push((0, -xyval[n][i].y + ylowhigh[n][i].x));
    }
    //write(dp);
    //write(dm);
    errorbars(xyval[n], dp, dm, graphpen);
    
    // Actualy plot things:
    draw(graph(xyval[n]), graphpen, legend, mark);
}

xaxis(xlabel, BottomTop, LeftTicks);

yaxis(ylabel, (secondary_filenames != "") ? Left : LeftRight,RightTicks);

// attach(legend(),point(plain.E),(((secondary_filenames != ""))
//                                ? 60*plain.E + 40 *plain.N
//                                 : 20*plain.E)  );
//attach(legend(),point(plain.S), N);
attach(legend(), point(S), 50*S);

if(secondary_filenames != "")
{
  write("secondary_filenames: ", secondary_filenames);
  string[] second_list = listfromcsv(secondary_filenames);
    
    datapoint[][] datapoints = new datapoint[second_list.length][];
    
    readfiles(second_list, datapoints);

    pair[][] xyval = new real[second_list.length][];
    pair[][] ylowhigh = new real[second_list.length][];
    for(int n = 0; n < datapoints.length; ++n) {
      datapoints_to_xyvallowhigh(datapoints[n], xyval[n], ylowhigh[n]);
    }
    
    bool interval = true;
    
    // // FIXME: speedup has error bounds, so we should read it, but
    // // p-vals does not.
    // readfiles(second_list, xyval, ylowhigh, true);

    picture secondarypic = secondaryY(new void(picture pic) {
	int penidx = testlist.length; // initialize to end of previous pen.

            scale(pic, xlog ? Log : Linear, Linear);
            
            for(int n = 0; n < xyval.length; ++n)
            {
                pen graphpen = Pen(penidx + n);
                if(penidx + n == 2) {
                    graphpen = darkgreen;
                }
                graphpen += dashed;
                
                guide g = scale(0.5mm) * unitcircle;
                marker mark = marker(g, Draw(graphpen + solid));
                        
                if(interval)
                {
                    // Compute the error bars:
                    pair[] dp;
                    pair[] dm;
                    for(int i = 0; i < xyval[n].length; ++i) {
                        dp.push((0, xyval[n][i].y - ylowhigh[n][i].x));
                        dm.push((0, xyval[n][i].y - ylowhigh[n][i].y));
                    }

                    errorbars(pic, xyval[n], dp, dm, graphpen);
		    
                }
                draw(pic,graph(pic, xyval[n]), graphpen, legends[n] + " vs " + legends[n+1],mark);
		write(xyval[n]);
		
		yequals(pic, 1.0, lightgrey);
		
            }

	    
            yaxis(pic, secondaryaxis, Right, black, LeftTicks);
	    attach(legend(pic), point(plain.E), 60*plain.E - 40 *plain.N  );
            //attach(legend(pic), point(plain.S), 120*S);
        });
    add(secondarypic);
}
