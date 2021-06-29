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
string secondaryaxis = "";

// Control whether inter-run speedup is plotted:
int speedup = 1;

// parameters for computing gflops and arithmetic intesntiy
real batchsize = 1;
real problemdim = 1;
real problemratio = 1;
bool realcomplex = false;
bool doubleprec = false;
string gpuid = "";
real bandwidth = 1228.8; // MI100

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
    // ROCFFT_LAYER=1 ./clients/staging/rocfft-rider --length $(asy -c "2^22") 2>&1| grep KERNEL |wc -l
    // Number of kernels for double-precision c2c 1D transforms.
    if (N <= 2^12)
        return 1.0;
    if (N <= 2^16)
        return 2.0;
    if (N <= 2^18)
        return 3.0;
    if (N <= 2^24)
        return 5.0;
    if (N <= 2^28)
        return 6.0;
    return 7.0;
}

// Compte the number of bytes read from and written to global memory for a transform.
real bytes(real Nx, real Ny, real Nz, real batch, bool realcomplex, bool doubleprec)
{
    real nvals = Nx * Ny * Nz * batch;
    // single or double precision, real or complex.
    real sizeof = (doubleprec ? 8 : 4) * (realcomplex ? 1 : 2);
    real nops = 2; // one read, one write.

    real bytes = nvals * sizeof * nops;
    
    if(just1dc2crad2) {
        // NB: only valid for c2c 1D transforms.
        bytes *= nkernels(Nx);
    }
    write("bytes:", Nx, Ny, Nz, batch, bytes);
    return bytes;
}

// Compute the number of FLOPs for a transform.
real flop(real Nx, real Ny, real Nz, real batch, bool realcomplex)
{
    real size = Nx * Ny * Nz;
    real fact = realcomplex ? 0.5 : 1.0;
    real flop = 5 * fact * batchsize * size * log(size) / log(2);
    return flop;
}

// Compute the performance in GFLOP/s.
// time in s, N is the problem size
real time2gflops(real t, real Nx, real Ny, real Nz, real batch, bool realcomplex)
{
    return 1e-9 * flop(Nx, Ny, Nz, batch, realcomplex) / t;
}


// Compute the performance in GFLOP/s.
// time in s, N is the problem size
real time2efficiency(real t, real Nx, real Ny, real Nz, real batch, bool realcomplex, bool doubleprec)
{
    real bw = bytes(Nx, Ny, Nz, batch, realcomplex, doubleprec) / t;
    return bw / (bandwidth * 1e9);
}


// Compute the arithmetic intensity for a transform.
real arithmeticintensity(real Nx, real Ny, real Nz, real batch, bool realcomplex,
                         bool doubleprec)
{
    return flop(Nx, Ny, Nz, batch, realcomplex) / bytes(Nx, Ny, Nz, batch, realcomplex, doubleprec);
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
real[][] Nx = new real[testlist.length][];
real[][] Ny = new real[testlist.length][];
real[][] Nz = new real[testlist.length][];
real[][][] data = new real[testlist.length][][];
real xmax = 0.0;
real xmin = inf;

// Read the data from the rocFFT-formatted data file.
void readfiles(string[] testlist, real[][] Nx, real[][] Ny, real[][] Nz, real[][][] data)
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
	    if(eof(fin))
	      {
                moretoread = false;
                break;

	      }
            if(dim == 0) {
                moretoread = false;
                break;
            }
            int xval = fin; // x-length
            int yval = (dim > 1) ? fin : 1;
            int zval = (dim > 2) ? fin : 1;        
            int nbatch = fin; // batch size
	    
            int N = fin; // Number of data points
            if (N > 0) {
	      real[] xvals = new real[N];
                for(int i = 0; i < N; ++i) {
		  xvals[i] = fin;
                }
		if(max(xvals) > 0.0) {
		  // Only add data if the data isn't all zero.
		  xmax = max(xval, xmax);
		  xmin = min(xval, xmin);
		  Nx[n].push(xval);
                  Ny[n].push((dim > 1) ? yval : 1);
                  Nz[n].push((dim > 2) ? zval : 1);
		  data[n][dataidx] = xvals;
		  ++dataidx;
		}
            }
        }
    }
}

readfiles(testlist, Nx, Ny, Nz, data);

// Plot the primary graph:
for(int n = 0; n < Nx.length; ++n)
{
    pen graphpen = Pen(n);
    if(n == 2)
        graphpen = darkgreen;
    string legend = myleg ? legends[n] : texify(testlist[n]);
    marker mark = marker(scale(0.5mm) * unitcircle, Draw(graphpen + solid));
    // Multi-axis graphs: set legend to appropriate y-axis.
    if(secondaryaxis != "")
        legend = "time";
    
    // We need to plot pairs for the error bars.
    pair[] z;
    pair[] dp;
    pair[] dm;

    bool drawme[] = new bool[Nx[n].length];
    for(int i = 0; i < drawme.length; ++i) {
        drawme[i] = true;
        if(!plotxval(Nx[n][i]))
	    drawme[i] = false;
    }

    // real[] horizvals:
    real[] xval;
    
    // y-values and bounds:
    real[] y;
    real[] ly;
    real[] hy;
    
    if(primaryaxis == "time") {
        xval = Nx[n];
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
        xval = Nx[n];
        for(int i = 0; i < data[n].length; ++i) {
            if(drawme[i]) {
                real[] vals;
                for(int j = 0; j < data[n][i].length; ++j) {
                    real val = time2gflops(data[n][i][j], Nx[n][i], Ny[n][i], Nz[n][i], batchsize, realcomplex);
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
        for(int i = 0; i < Nx[n].length; ++i) {
            xval.push(arithmeticintensity(Nx[n][i], Ny[n][i], Nz[n][i], batchsize,
                                          realcomplex, doubleprec));
        }
        for(int i = 0; i < data[n].length; ++i) {
            if(drawme[i]) {
                real[] vals;
                for(int j = 0; j < data[n][i].length; ++j) {
                    real val = time2gflops(data[n][i][j], Nx[n][i], Ny[n][i], Nz[n][i],
                                           batchsize, realcomplex);
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
                    real p = log(Nx[n][i]) / log(2);
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
    yaxis(ylabel,(speedup > 1 || (secondaryaxis != "")) ? Left : LeftRight,RightTicks);
else
    yaxis(ylabel,LeftRight);

if(dolegend)
    attach(legend(),point(plain.E),((speedup > 1  || (secondaryaxis != ""))
                                    ? 60*plain.E + 40 *plain.N
                                    : 20*plain.E)  );

// Add a secondary axis showing speedup.
if(speedup > 1) {
    string[] legends = listfromcsv(legendlist);
    // TODO: when there is data missing at one end, the axes might be weird

    picture secondarypic = secondaryY(new void(picture pic) {
            scale(pic,Log,Linear);
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
		  
                    for(int i = 0; i < Nx[n].length; ++i) {
                        for(int j = 0; j < Nx[n+next].length; ++j) {
                            if (Nx[n][i] == Nx[n+next][j]) {
                                baseval.push(Nx[n][i]);
                                real yni = getmedian(data[n][i]);
                                real ynextj = getmedian(data[n+next][j]);
                                real val = yni / ynextj;
                                yval.push(val);

                                zy.push((Nx[n][i], val));
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

			if(ymin == inf || ymin == -inf)
                            ymin = 1;
			if(ymax == inf || ymax == -inf)
                            ymax = 1;
			real[] fakey = {ymin, ymax};
			write(fakex);
			write(fakey);
			
                        // draw an invisible graph to set up the limits correctly.
                        draw(pic,graph(pic,fakex, fakey),invisible);

                    }
                }
            }

	    yequals(pic, 1.0, lightgrey);
            yaxis(pic, "speedup", Right, black, LeftTicks);
            attach(legend(pic),point(plain.E), 60*plain.E - 40 *plain.N  );
        });
    
    add(secondarypic);
}

// Add a secondary axis showing GFLOP/s.
if(secondaryaxis == "gflops") {
    string[] legends = listfromcsv(legendlist);
    picture secondaryG = secondaryY(new void(picture pic) {
	    //int penidx = testlist.length;
            scale(pic, Log(true), Log(true));
            for(int n = 0; n < Nx.length; ++n) {

                pen graphpen = Pen(n+1);
                if(n == 2)
                    graphpen = darkgreen;
                graphpen += dashed;
                
                real[] y = new real[];
                real[] ly = new real[];
                real[] hy = new real[];
                for(int i = 0; i < data[n].length; ++i) {
                    write(Nx[n][i]);
                    real[] gflops = new real[];
                    for(int j = 0; j < data[n][i].length; ++j) {
                        real val = time2gflops(data[n][i][j],
                                               Nx[n][i],
                                               Ny[n][i],
                                               Nz[n][i],
                                               batchsize,
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
                draw(pic, graph(pic, Nx[n], y), graphpen, legend = texify("GFLOP/s"), mark);
                
                pair[] z = new pair[];
                pair[] dp = new pair[];
                pair[] dm = new pair[];
                for(int i = 0; i < Nx[n].length; ++i) {
                    z.push((Nx[n][i] , y[i]));
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

// Add a secondary axis showing efficiency.
if(secondaryaxis == "efficiency") {
    string[] legends = listfromcsv(legendlist);
    picture secondaryG = secondaryY(new void(picture pic) {
	    //int penidx = testlist.length;
            scale(pic, Log(true), Linear);
            for(int n = 0; n < Nx.length; ++n) {

                pen graphpen = Pen(n+1);
                if(n == 2)
                    graphpen = darkgreen;
                graphpen += dashed;
                
                real[] y = new real[];
                real[] ly = new real[];
                real[] hy = new real[];
                for(int i = 0; i < data[n].length; ++i) {
                    real[] vals = new real[];
                    for(int j = 0; j < data[n][i].length; ++j) {
                        real val = time2efficiency(data[n][i][j],
                                                   Nx[n][i],
                                                   Ny[n][i],
                                                   Nz[n][i],
                                                   batchsize,
                                                   realcomplex,
                                                   doubleprec);
                        vals.push(val);
                    }
                    real[] medlh = mediandev(vals);
                    y.push(medlh[0]);
                    ly.push(medlh[1]);
                    hy.push(medlh[2]);
                }
                guide g = scale(0.5mm) * unitcircle;
                marker mark = marker(g, Draw(graphpen + solid));
                
                draw(pic, graph(pic, Nx[n], y), graphpen, legend = texify("efficiency"), mark);
                
                pair[] z = new pair[];
                pair[] dp = new pair[];
                pair[] dm = new pair[];
                for(int i = 0; i < Nx[n].length; ++i) {
                    z.push((Nx[n][i] , y[i]));
                    dp.push((0 , y[i] - hy[i]));
                    dm.push((0 , y[i] - ly[i]));
                }
                errorbars(pic, z, dp, dm, graphpen);
            }
                yaxis(pic, "Efficiency", Right, black, LeftTicks(DefaultFormat, begin=true, end=true), ymin=-infinity, ymax=infinity);

            attach(legend(pic), point(plain.E), 60*plain.E - 40 *plain.N  );
    
        });

    add(secondaryG);
}
