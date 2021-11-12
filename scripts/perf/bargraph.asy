import graph;
import utils;

struct bardata {
  string label;
  real y;
  real ylow;
  real yhigh;
}

void readbarfiles(string[] filelist, bardata[][] data)
{
    for(int n = 0; n < filelist.length; ++n)
    {
      data.push(new bardata[]);
      
      string filename = filelist[n];
//      write(filename);
      
      file fin = input(filename).line().word();
      string[] hdr = fin;

      while(!eof(fin)) {
        real elements;
      
	bardata dat;
	dat.label = fin;
        elements = fin;
	dat.y = (real)fin;
	dat.ylow = (real)fin;
	dat.yhigh = (real)fin;
	
//	write(dat.label, dat.y, dat.ylow, dat.yhigh);

	data[n].push(dat);

      }
    }
//    write("done reading");
}

void drawbargraph(bardata[][] data, string[] legs, string[] otherlegs) {
  // Let's go to the bar, eh?

  // Assumption: same number of data points.

  int nbars = data.length;

  real width = 1.0 / nbars;
  real skip = 0.5;
  
  // Loop through all the data sets.
  for(int n = 0; n < nbars; ++n) {
    pen p = Pen(n); // + opacity(0.5);

    int len = data[n].length;

    // Set up the left and right sides of the bars.
    real[] left = new real[len];
    real[] right = new real[len];
    left[0] = n * width;
    for(int i = 1; i < len; ++i) {
      left[i] = left[i - 1] + nbars * width + skip;
    }
    for(int i = 0; i < len; ++i) {
      right[i] = left[i] + width;
    }
    
    // Draw an invisible graph to set up the axes.
    real[] fakex = new real[len];
    fakex[0] = left[0];
    for(int i = 1; i < fakex.length; ++i) {
      fakex[i] = right[i];
    }
    real[] yvals = new real[len];
    for(int i = 0; i < len; ++i) {
      yvals[i] = data[n][i].y;
    }
    draw(graph(left, yvals), invisible, legend = Label(otherlegs[n], p)); 


    // TOTO: in log plots, compute a better bottom.
    real bottom = 0.0;
    
    // Draw the bars
    for(int i = 0; i < data[n].length; ++i) {
      pair p0 = Scale((left[i], data[n][i].y));
      pair p1 = Scale((right[i], data[n][i].y));
      pair p2 = Scale((right[i], bottom));
      pair p3 = Scale((left[i], bottom));
      filldraw(p0--p1--p2--p3--cycle, p, black);
    }

   
    // Draw the bounds:
    for(int i = 0; i < data[n].length; ++i) {
      real xval = 0.5 * (left[i] + right[i]);
      pair plow = (xval, data[n][i].ylow);
      dot(plow);
      pair phigh = (xval, data[n][i].yhigh);
      dot(phigh);
      draw(plow--phigh);
      draw(plow-(0.25*width)--plow+(0.25*width));
      draw(phigh-(0.25*width)--phigh+(0.25*width));
    }

    
    // This is there the legends go
    if(n == nbars - 1) {
      for(int i = 0; i <  data[n].length; ++i) {
	pair p = (0.5 * nbars * width + i * (skip + nbars * width), 0);
	// 	//label(rotate(90) * Label(xleg[i]), align=S, p);
	label(Label(data[n][i].label), align=S, p);
      }
    }
    
  }
}

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

// Input data:
string filenames = "";
string secondary_filenames = "";
string legendlist = "";

// Graph formatting
string xlabel = "Problem size type";
string ylabel = "Time [s]";

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

// TODO: the first column will eventually be text.
string[] testlist = listfromcsv(filenames);

// Data containers:
pair[][] xyval = new real[testlist.length][];
pair[][] ylowhigh = new real[testlist.length][];


bardata[][] data;

readbarfiles(testlist, data);

// for(int n = 0; n < data.length; ++n) {
//   for(int i = 0; i < data[n].length; ++i) {
//     write(data[n][i].label, data[n][i].y, data[n][i].ylow, data[n][i].yhigh);
//   }
// }

//write(ylowhigh);

//write(xyval);

// Generate bar legends.
string[] legs = {};
for(int i = 0; i < xyval[0].length; ++i) {
  legs.push(string(xyval[0][i].x));
}

drawbargraph(data, legs, legends);

xaxis(BottomTop);
yaxis("Time (s)", LeftRight, RightTicks);

attach(legend(),point(plain.E),  20*plain.E);

