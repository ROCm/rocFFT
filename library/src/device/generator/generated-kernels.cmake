# The following is a list of known generated kernels.  This list is sufficient for now, but really should be
# query-able from the executable itself.  An idea would be to have the generator executable accept a CLI
# flag that dumps all generated filenames into an output.txt file.  CMake would then slurp that into
# a cmake list.

set( kernels_pow2
rocfft_kernel_1024.h
rocfft_kernel_128.h
rocfft_kernel_128_sbcc.h
rocfft_kernel_128_sbrc.h
rocfft_kernel_16.h
rocfft_kernel_1.h
rocfft_kernel_2048.h
rocfft_kernel_256.h
rocfft_kernel_256_sbcc.h
rocfft_kernel_256_sbrc.h
rocfft_kernel_2.h
rocfft_kernel_32.h
rocfft_kernel_4096.h
rocfft_kernel_4.h
rocfft_kernel_512.h
rocfft_kernel_64.h
rocfft_kernel_64_sbcc.h
rocfft_kernel_64_sbrc.h
rocfft_kernel_8.h
)

set( kernels_pow3
rocfft_kernel_1.h
rocfft_kernel_2187.h
rocfft_kernel_243.h
rocfft_kernel_27.h
rocfft_kernel_3.h
rocfft_kernel_729.h
rocfft_kernel_81.h
rocfft_kernel_9.h
)

set( kernels_pow5
rocfft_kernel_125.h
rocfft_kernel_1.h
rocfft_kernel_25.h
rocfft_kernel_3125.h
rocfft_kernel_5.h
rocfft_kernel_625.h
)

set( kernels_all
rocfft_kernel_1000.h
rocfft_kernel_100.h
rocfft_kernel_1024.h
rocfft_kernel_1080.h
rocfft_kernel_108.h
rocfft_kernel_10.h
rocfft_kernel_1125.h
rocfft_kernel_1152.h
rocfft_kernel_1200.h
rocfft_kernel_120.h
rocfft_kernel_1215.h
rocfft_kernel_1250.h
rocfft_kernel_125.h
rocfft_kernel_1280.h
rocfft_kernel_128.h
rocfft_kernel_128_sbcc.h
rocfft_kernel_128_sbrc.h
rocfft_kernel_1296.h
rocfft_kernel_12.h
rocfft_kernel_1350.h
rocfft_kernel_135.h
rocfft_kernel_1440.h
rocfft_kernel_144.h
rocfft_kernel_1458.h
rocfft_kernel_1500.h
rocfft_kernel_150.h
rocfft_kernel_1536.h
rocfft_kernel_15.h
rocfft_kernel_1600.h
rocfft_kernel_160.h
rocfft_kernel_1620.h
rocfft_kernel_162.h
rocfft_kernel_16.h
rocfft_kernel_1728.h
rocfft_kernel_1800.h
rocfft_kernel_180.h
rocfft_kernel_1875.h
rocfft_kernel_18.h
rocfft_kernel_1920.h
rocfft_kernel_192.h
rocfft_kernel_1944.h
rocfft_kernel_1.h
rocfft_kernel_2000.h
rocfft_kernel_200.h
rocfft_kernel_2025.h
rocfft_kernel_2048.h
rocfft_kernel_20.h
rocfft_kernel_2160.h
rocfft_kernel_216.h
rocfft_kernel_2187.h
rocfft_kernel_2250.h
rocfft_kernel_225.h
rocfft_kernel_2304.h
rocfft_kernel_2400.h
rocfft_kernel_240.h
rocfft_kernel_2430.h
rocfft_kernel_243.h
rocfft_kernel_24.h
rocfft_kernel_2500.h
rocfft_kernel_250.h
rocfft_kernel_2560.h
rocfft_kernel_256.h
rocfft_kernel_256_sbcc.h
rocfft_kernel_256_sbrc.h
rocfft_kernel_2592.h
rocfft_kernel_25.h
rocfft_kernel_2700.h
rocfft_kernel_270.h
rocfft_kernel_27.h
rocfft_kernel_2880.h
rocfft_kernel_288.h
rocfft_kernel_2916.h
rocfft_kernel_2.h
rocfft_kernel_3000.h
rocfft_kernel_300.h
rocfft_kernel_3072.h
rocfft_kernel_30.h
rocfft_kernel_3125.h
rocfft_kernel_3200.h
rocfft_kernel_320.h
rocfft_kernel_3240.h
rocfft_kernel_324.h
rocfft_kernel_32.h
rocfft_kernel_3375.h
rocfft_kernel_3456.h
rocfft_kernel_3600.h
rocfft_kernel_360.h
rocfft_kernel_3645.h
rocfft_kernel_36.h
rocfft_kernel_3750.h
rocfft_kernel_375.h
rocfft_kernel_3840.h
rocfft_kernel_384.h
rocfft_kernel_3888.h
rocfft_kernel_3.h
rocfft_kernel_4000.h
rocfft_kernel_400.h
rocfft_kernel_4050.h
rocfft_kernel_405.h
rocfft_kernel_4096.h
rocfft_kernel_40.h
rocfft_kernel_432.h
rocfft_kernel_450.h
rocfft_kernel_45.h
rocfft_kernel_480.h
rocfft_kernel_486.h
rocfft_kernel_48.h
rocfft_kernel_4.h
rocfft_kernel_500.h
rocfft_kernel_50.h
rocfft_kernel_512.h
rocfft_kernel_540.h
rocfft_kernel_54.h
rocfft_kernel_576.h
rocfft_kernel_5.h
rocfft_kernel_600.h
rocfft_kernel_60.h
rocfft_kernel_625.h
rocfft_kernel_640.h
rocfft_kernel_648.h
rocfft_kernel_64.h
rocfft_kernel_64_sbcc.h
rocfft_kernel_64_sbrc.h
rocfft_kernel_675.h
rocfft_kernel_6.h
rocfft_kernel_720.h
rocfft_kernel_729.h
rocfft_kernel_72.h
rocfft_kernel_750.h
rocfft_kernel_75.h
rocfft_kernel_768.h
rocfft_kernel_800.h
rocfft_kernel_80.h
rocfft_kernel_810.h
rocfft_kernel_81.h
rocfft_kernel_864.h
rocfft_kernel_8.h
rocfft_kernel_900.h
rocfft_kernel_90.h
rocfft_kernel_960.h
rocfft_kernel_96.h
rocfft_kernel_972.h
rocfft_kernel_9.h
)

set( kernels_launch
kernel_launch_generator.h
function_pool.cpp.h
function_pool.cpp
kernel_launch_single_large.cpp.h
kernel_launch_single_large.cpp
kernel_launch_double_large.cpp.h
kernel_launch_double_large.cpp
)

set( small_kernels_group_num 8 )
MATH(EXPR max_group_num "${small_kernels_group_num}-1")
foreach(small_kernel_group_id RANGE 0 ${max_group_num} )
    string(CONCAT small_kernel_single_file "kernel_launch_single_" ${small_kernel_group_id} ".cpp")
    string(CONCAT small_kernel_single_h_file "kernel_launch_single_" ${small_kernel_group_id} ".cpp.h")
    string(CONCAT small_kernel_double_file "kernel_launch_double_" ${small_kernel_group_id} ".cpp")
    string(CONCAT small_kernel_double_h_file "kernel_launch_double_" ${small_kernel_group_id} ".cpp.h")

    list(APPEND kernels_launch ${small_kernel_single_file} ${small_kernel_single_h_file})
    list(APPEND kernels_launch ${small_kernel_double_file} ${small_kernel_double_h_file})
endforeach(small_kernel_group_id)

