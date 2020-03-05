#!/bin/bash
# Known Limitations:
#     This script must be started from the directory it is located in!
#     Coarse grained repetitions (-n) are not compatible with plotting/document generation
#     Fine grained repetitions (-N) are not actually passed to the executable. The value is hardcoded to 10.
#
# Other Notes:
#     The same input directory can be specified multiple times
#     The output directory must be unique. It is an error to recycle it.
#     If labels are recycled, the intent is to combine the results from the respective output directories, but there is only partial support for this.
rocfft_build=../../build/release/clients/staging/

# The only required argument is the input directory
# Always overwrites any existing results
./alltime.py -i $rocfft_build

# Expand the scope of the sweep using -R,
# Use -m to ensure that the default runs are re-used
# Use -w to specify a different output document
./alltime.py -i $rocfft_build \
             -m EXECUTE PLOT DOCUMENT -R report benchmark efficiency \
             -w doc_all_run_types

# Use only the 1D tests from the report and create a new report
./alltime.py -i $rocfft_build \
             -m PLOT DOCUMENT \
             -D 1 \
             -R report \
             -w doc_1d_report

# Add the output and input directories to compare against the same run
./alltime.py -i $rocfft_build \
             -i $rocfft_build \
             -o /tmp/bench_test_dir0 -o /tmp/bench_test_dir1 -l run1 -l "run repeat" \
             -D 2 3 -s \
             -R report \
             -w doc_self_compare

# Same as previous, but add speed-up plots
./alltime.py -i $rocfft_build \
             -i $rocfft_build \
             -o /tmp/bench_test_dir0 -o /tmp/bench_test_dir1 -l run1 -l "run repeat" \
             -D 2 3 -s \
             -R report \
             -m PLOT DOCUMENT \
             --speedup \
             -w doc_self_compare_with_speedup

# Look at the 3D raw data from the first run interactively assuming matplotlib is installed.
# If matplotlib is not installed, do nothing.
./alltime.py -i $rocfft_build \
             -m INTERACTIVE \
             -D 3
