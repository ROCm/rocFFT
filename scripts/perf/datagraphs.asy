import graph;
import utils;
import stats;

//asy datagraphs -u "xlabel=\"\$\\bm{u}\cdot\\bm{B}/uB$\"" -u "doyticks=false" -u "ylabel=\"\"" -u "legendlist=\"a,b\""

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

//scale(Linear,Linear);
scale(Log, Log);
//scale(Log,Linear);

// Plot bounds
real xmin = -inf;
real xmax = inf;

bool dolegend = true;

// Input data:
string filenames = "";
string legendlist = "";

// Graph formatting
bool doxticks = true;
bool doyticks = true;
string xlabel = "Problem size N";
string ylabel = "Time [s]";
bool normalize = false;

bool just1dc2crad2 = false;

string primaryaxis = "time";

// Control whether inter-run speedup is plotted:
int speedup = 1;

bool secondarygflops = false;
// parameters for computing gflops and arithmetic intesntiy
real batchsize = 1;
real problemdim = 1;
real problemratio = 1;
bool realcomplex = false;
bool doubleprec = false;
string gpuid = "";

usersetting();

if(primaryaxis == "gflops")
    ylabel = "GFLOP/s";

if(primaryaxis == "roofline") {
    ylabel = "GFLOP/s";
    xlabel = "arithmetic intensity";
    scale(Linear,Log);
}

write("filenames:\"", filenames+"\"");
if(filenames == "")
    filenames = getstring("filenames");

if (legendlist == "")
    legendlist = filenames;
bool myleg = ((legendlist == "") ? false : true);
string[] legends = set_legends(legendlist);
for (int i = 0; i < legends.length; ++i) {
  legends[i] = texify(legends[i]);
}
  
if(normalize) {
   scale(Log, Linear);
   ylabel = "Time / problem size N, [s]";
}

bool plotxval(real x) {
   return x >= xmin && x <= xmax;
}

real nkernels(real N)
{
    // rocfft-rider-d  --length $(asy -c "2^27") | grep KERNEL | wc -l
    // Number of kernels for c2c 1D transforms.
    if (N <= 2^11)
        return 1.0;
    if (N <= 2^15)
        return 2.0;
    if (N <= 2^17)
        return 3.0;
    if (N <= 2^22)
        return 5.0;
    if (N <= 2^26)
        return 6.0;
    return 7.0;
}

// Compte the number of bytes read from and written to global memory for a transform.
real bytes(real N, real batch, real dim, real ratio, bool realcomplex, bool doubleprec)
{
    real bytes = batch * 4 * N^dim * ratio; // read and write one complex arrays of length N.
    if(realcomplex)
        bytes /= 2;
    if(doubleprec)
        bytes *= 8;
    else
        bytes *= 4;

    if(just1dc2crad2) {
        // NB: only valid for c2c 1D transforms.
        bytes *= nkernels(N);
    }
    
    return bytes;
}

// Compute the number of FLOPs for a transform.
real flop(real N, real batch, real dim, real ratio, bool realcomplex)
{
    real size = ratio * N^dim;
    real fact = realcomplex ? 0.5 : 1.0;
    real flop = 5 * fact * batchsize * size * log(size) / log(2);
    return flop;
}

// Compute the performance in GFLOP/s.
// time in s, N is the problem size
real time2gflops(real t, real N, real batch, real dim, real ratio,
                 bool realcomplex)
{
    return 1e-9 *  flop(N, batch, dim, ratio, realcomplex) / t;
}

// Compute the arithmetic intensity for a transform.
real arithmeticintensity(real N, real batch, real dim, real ratio, bool realcomplex,
                         bool doubleprec)
{
    return flop(N, batch, dim, ratio, realcomplex)
        / bytes(N, batch, dim, ratio, realcomplex, doubleprec);
}

// Create an array from a comma-separated string
string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}

string[] testlist = listfromcsv(filenames);

// Data containers:
real[][] x = new real[testlist.length][];
real[][][] data = new real[testlist.length][][];
real xmax = 0.0;
real xmin = inf;

// Read the data from the rocFFT-formatted data file.
void readfiles(string[] testlist, real[][] x, real[][][] data)
{
// Get the data from the file:
    for(int n = 0; n < testlist.length; ++n)
    {
        string filename = testlist[n];
        write(filename);
        data[n] = new real[][];

        int dataidx = 0;
        bool moretoread = true;
        file fin = input(filename);
        while(moretoread) {
            int dim = fin; // Problem dimension
            if(dim == 0) {
                moretoread = false;
                break;
            }
            int xval = fin; // x-length
            if(dim > 1) {
                int yval = fin; // y-length
            }
            if(dim > 2) {
                int zval = fin; // z-length
            }
            int nbatch = fin; // batch size
            
            int N = fin; // Number of data points
            if (N > 0) {
                xmax = max(xval, xmax);
                xmin = min(xval, xmin);
                x[n].push(xval);
                data[n][dataidx] = new real[N];
                for(int i = 0; i < N; ++i) {
                    data[n][dataidx][i] = fin;
                }
                ++dataidx;
            }
        }
    }
}

readfiles(testlist, x, data);

// Plot the primary graph:
for(int n = 0; n < x.length; ++n)
{
    pen graphpen = Pen(n);
    if(n == 2)
        graphpen = darkgreen;
    string legend = myleg ? legends[n] : texify(testlist[n]);
    marker mark = marker(scale(0.5mm) * unitcircle, Draw(graphpen + solid));
    // Multi-axis graphs: set legend to appropriate y-axis.
    if(secondarygflops)
        legend = "time";
    
    // We need to plot pairs for the error bars.
    pair[] z;
    pair[] dp;
    pair[] dm;

    bool drawme[] = new bool[x[n].length];
    for(int i = 0; i < drawme.length; ++i) {
        drawme[i] = true;
        if(!plotxval(x[n][i]))
	    drawme[i] = false;
    }

    // real[] horizvals:
    real[] xval;
    
    // y-values and bounds:
    real[] y;
    real[] ly;
    real[] hy;
    
    if(primaryaxis == "time") {
        xval = x[n];
        for(int i = 0; i < data[n].length; ++i) {
            if(drawme[i]) {
                real[] medlh = mediandev(data[n][i]);
                y.push(medlh[0]);
                ly.push(medlh[1]);
                hy.push(medlh[2]);
        
                z.push((xval[i] , y[i]));
                dp.push((0 , y[i] - hy[i]));
                dm.push((0 , y[i] - ly[i]));
            }
        }
    }
    
    if(primaryaxis == "gflops") {
        xval = x[n];
        for(int i = 0; i < data[n].length; ++i) {
            if(drawme[i]) {
                real[] vals;
                for(int j = 0; j < data[n][i].length; ++j) {
                    real val = time2gflops(data[n][i][j], x[n][i],
                                           batchsize, problemdim, problemratio, realcomplex);
                    //write(val);
                    vals.push(val);
                }
                real[] medlh = mediandev(vals);
                y.push(medlh[0]);
                ly.push(medlh[1]);
                hy.push(medlh[2]);
                    
                z.push((xval[i] , y[i]));
                dp.push((0 , y[i] - hy[i]));
                dm.push((0 , y[i] - ly[i]));
            }
        }
    }

    if(primaryaxis == "roofline") {
        for(int i = 0; i < x[n].length; ++i) {
            xval.push(arithmeticintensity(x[n][i], batchsize, problemdim, problemratio,
                                          realcomplex, doubleprec));
        }
        for(int i = 0; i < data[n].length; ++i) {
            if(drawme[i]) {
                real[] vals;
                for(int j = 0; j < data[n][i].length; ++j) {
                    real val = time2gflops(data[n][i][j], x[n][i],
                                           batchsize, problemdim, problemratio, realcomplex);
                    //write(val);
                    vals.push(val);
                }
                real[] medlh = mediandev(vals);
                y.push(medlh[0]);
                ly.push(medlh[1]);
                hy.push(medlh[2]);

                z.push((xval[i] , y[i]));
                dp.push((0 , y[i] - hy[i]));
                dm.push((0 , y[i] - ly[i]));
            }
        }
    }

    // write(xval);
    // write(y);
    
    // Actualy plot things:
    errorbars(z, dp, dm, graphpen);
    draw(graph(xval, y, drawme), graphpen, legend, mark);
    
    if(primaryaxis == "roofline") {
        real bw = 0; // device bandwidth in GB/s
        real maxgflops = 0; // max device speed in GFLOP/s

        if(just1dc2crad2) {
            int skip = z.length > 8 ? 2 : 1;
            for(int i = 0; i < z.length; ++i) {
                //dot(Scale(z[i]));
                //dot(Label("(3,5)",align=S),Scale(z));
                if(i % skip == 0) {
                    real p = log(x[n][i]) / log(2);
                    label("$2^{"+(string)p+"}$",Scale(z[i]),S);
                }
            }
        }
        
        if(gpuid == "0x66af") {
            // Radeon7
            // https://www.amd.com/en/products/graphics/amd-radeon-vii
            bw = 1024;
            maxgflops = 1000 * (doubleprec ? 3.46 : 13.8);
        }
        if(gpuid == "0x66a1") {
            // mi60
            // https://www.amd.com/system/files/documents/radeon-instinct-mi60-datasheet.pdf
            bw = 1024;
            maxgflops = 1000 * (doubleprec ? 7.4 : 14.7);
        }
        
        if(bw > 0 && maxgflops > 0) {
            real aistar = maxgflops / bw;
            real a = min(xval);
            real b = max(xval);
            if(aistar < a) {
                // Always compute limited.
                yequals(maxgflops, grey);
            }
            else if(aistar > b) {
                // Always bandwidth limited
                pair[] roofline = {(a, a * bw), (b, b * bw)};
                draw(graph(roofline), grey);
            } else {
                // General case.
                pair[] roofline = {(a, a * bw), (aistar, aistar * bw), (b, maxgflops)};
                draw(graph(roofline), grey);
            }
             
        }
        // TODO: fix y-axis bound.
    }
}

if(doxticks)
   xaxis(xlabel,BottomTop,LeftTicks);
else
   xaxis(xlabel);

if(doyticks)
    yaxis(ylabel,(speedup > 1 || secondarygflops) ? Left : LeftRight,RightTicks);
else
   yaxis(ylabel,LeftRight);

if(dolegend)
    attach(legend(),point(plain.E),((speedup > 1  || secondarygflops)
                                    ? 60*plain.E + 40 *plain.N
                                    : 20*plain.E)  );

// Add a secondary axis showing speedup.
if(speedup > 1) {
    string[] legends = listfromcsv(legendlist);
    // TODO: when there is data missing at one end, the axes might be weird

    picture secondary = secondaryY(new void(picture pic) {
            scale(pic,Log,Log);
            real ymin = inf;
            real ymax = -inf;
	    int penidx = testlist.length;
            for(int n = 0; n < testlist.length; n += speedup) {

                for(int next = 1; next < speedup; ++next) {
                    real[] baseval = new real[];
                    real[] yval = new real[];
                    pair[] zy;
                    pair[] dp;
                    pair[] dm;
		  
                    for(int i = 0; i < x[n].length; ++i) {
                        for(int j = 0; j < x[n+next].length; ++j) {
                            if (x[n][i] == x[n+next][j]) {
                                baseval.push(x[n][i]);
                                real yni = getmedian(data[n][i]);
                                real ynextj = getmedian(data[n+next][j]);
                                real val = yni / ynextj;
                                yval.push(val);

                                zy.push((x[n][i], val));
                                real[] lowhi = ratiodev(data[n][i], data[n+next][j]);
                                real hi = lowhi[1];
                                real low = lowhi[0];

                                dp.push((0 , hi - val));
                                dm.push((0 , low - val));
    
                                ymin = min(val, ymin);
                                ymax = max(val, ymax);
                                break;
                            }
                        }
                    }

		  
                    if(baseval.length > 0){
                        pen p = Pen(penidx)+dashed;
                        ++penidx;
		  
                        guide g = scale(0.5mm) * unitcircle;
                        marker mark = marker(g, Draw(p + solid));
                
                        draw(pic,graph(pic,baseval, yval),p,legends[n] + " vs " + legends[n+next],mark);
                        errorbars(pic, zy, dp, dm, p);
                    }

                    {
                        real[] fakex = {xmin, xmax};
                        real[] fakey = {ymin, ymax};
                        // draw an invisible graph to set up the limits correctly.
                        draw(pic,graph(pic,fakex, fakey),invisible);

                    }
                }
            }

	    yequals(pic, 1.0, lightgrey);
            yaxis(pic, "speedup", Right, black, LeftTicks);
            attach(legend(pic),point(plain.E), 60*plain.E - 40 *plain.N  );
        });

    add(secondary);
}

// Add a secondary axis showing GFLOP/s.
if(secondarygflops) {
    string[] legends = listfromcsv(legendlist);
    picture secondaryG=secondaryY(new void(picture pic) {
	    //int penidx = testlist.length;
            scale(pic, Log(true), Log(true));
            for(int n = 0; n < x.length; ++n) {

                pen graphpen = Pen(n+1);
                if(n == 2)
                    graphpen = darkgreen;
                graphpen += dashed;
                
                real[] y = new real[];
                real[] ly = new real[];
                real[] hy = new real[];
                for(int i = 0; i < data[n].length; ++i) {
                    write(x[n][i]);
                    real[] gflops = new real[];
                    for(int j = 0; j < data[n][i].length; ++j) {
                        real val = time2gflops(data[n][i][j],
                                               x[n][i],
                                               batchsize,
                                               problemdim,
                                               problemratio,
                                               realcomplex);
                        write(val);
                        gflops.push(val);
                    }
                    real[] medlh = mediandev(gflops);
                    y.push(medlh[0]);
                    ly.push(medlh[1]);
                    hy.push(medlh[2]);
                }
                guide g = scale(0.5mm) * unitcircle;
                marker mark = marker(g, Draw(graphpen + solid));
                draw(pic, graph(pic, x[n], y), graphpen, legend = texify("GFLOP/s"), mark);
                
                pair[] z = new pair[];
                pair[] dp = new pair[];
                pair[] dm = new pair[];
                for(int i = 0; i < x[n].length; ++i) {
                    z.push((x[n][i] , y[i]));
                    dp.push((0 , y[i] - hy[i]));
                    dm.push((0 , y[i] - ly[i]));
                }
                errorbars(pic, z, dp, dm, graphpen);
            }
            yaxis(pic, "GFLOP/s", Right, black, LeftTicks(begin=false,end=false));

            attach(legend(pic), point(plain.E), 60*plain.E - 40 *plain.N  );
    
        });

    add(secondaryG);
 }


