cmake_minimum_required( VERSION 2.8.12 )

set (CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set (CMAKE_CXX_FLAGS "-I .")

set(benchmark_fft_source
    main.cpp launch.cpp twiddles.cpp )

add_executable( benchmark_fft ${benchmark_fft_source} )


