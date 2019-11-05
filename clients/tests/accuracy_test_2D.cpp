// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <algorithm>
#include <array>
#include <complex>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

// Set parameters

// TODO: enable 16384, 32768 when omp support is available (takes too
// long!)
static std::vector<size_t> pow2_range = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

// TODO: 2187 and 6561 fail with the new test infrastructure.
static std::vector<size_t> pow3_range = {3, 27, 81, 243, 729};

// TODO: 3125 and 15625 fail with the new test infrastructure.
static std::vector<size_t> pow5_range = {5, 25, 125, 625};

static std::vector<size_t> prime_range = {7, 11, 13, 17, 19, 23, 29, 263, 269, 271, 277};

static size_t batch_range[] = {1, 2};

static size_t stride_range[] = {1};

static rocfft_result_placement placeness_range[]
    = {rocfft_placement_notinplace, rocfft_placement_inplace};

static rocfft_transform_type transform_range[]
    = {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse};

static data_pattern pattern_range[] = {sawtooth};

// Test suite classes:

class accuracy_test_complex_2D : public ::testing::TestWithParam<std::tuple<size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            rocfft_result_placement,
                                                                            size_t,
                                                                            data_pattern,
                                                                            rocfft_transform_type>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};
class accuracy_test_real_2D
    : public ::testing::TestWithParam<
          std::tuple<size_t, size_t, size_t, rocfft_result_placement, size_t, data_pattern>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

//  Complex to complex
template <typename Tfloat>
void normal_2D_complex_interleaved_to_complex_interleaved(size_t                  Nx,
                                                          size_t                  Ny,
                                                          size_t                  batch,
                                                          rocfft_result_placement placeness,
                                                          rocfft_transform_type   transform_type,
                                                          size_t                  stride,
                                                          data_pattern            pattern)
{
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;

    std::vector<size_t> length = {Ny, Nx};

    const bool inplace = placeness == rocfft_placement_inplace;
    int        sign
        = transform_type == rocfft_transform_type_complex_forward ? FFTW_FORWARD : FFTW_BACKWARD;

    // Dimension configuration:
    std::array<fftw_iodim64, 2> dims;
    dims[1].n  = Ny;
    dims[1].is = stride;
    dims[1].os = stride;
    dims[0].n  = Nx;
    dims[0].is = dims[1].n * dims[1].is;
    dims[0].os = dims[1].n * dims[1].os;

    // Batch configuration:
    std::array<fftw_iodim64, 1> howmany_dims;
    howmany_dims[0].n  = batch;
    howmany_dims[0].is = dims[0].n * dims[0].is;
    howmany_dims[0].os = dims[0].n * dims[0].os;

    const size_t isize = howmany_dims[0].n * howmany_dims[0].is;
    const size_t osize = howmany_dims[0].n * howmany_dims[0].os;

    if(verbose)
    {
        if(inplace)
            std::cout << "in-place\n";
        else
            std::cout << "out-of-place\n";
        for(int i = 0; i < dims.size(); ++i)
        {
            std::cout << "dim " << i << std::endl;
            std::cout << "\tn: " << dims[i].n << std::endl;
            std::cout << "\tis: " << dims[i].is << std::endl;
            std::cout << "\tos: " << dims[i].os << std::endl;
        }
        std::cout << "howmany_dims:\n";
        std::cout << "\tn: " << howmany_dims[0].n << std::endl;
        std::cout << "\tis: " << howmany_dims[0].is << std::endl;
        std::cout << "\tos: " << howmany_dims[0].os << std::endl;
        std::cout << "isize: " << isize << "\n";
        std::cout << "osize: " << osize << "\n";
    }

    // Set up buffers:
    // Local data buffer:
    std::complex<Tfloat>* cpu_in = fftw_alloc_type<std::complex<Tfloat>>(isize);
    // Output buffer
    std::complex<Tfloat>* cpu_out = inplace ? cpu_in : fftw_alloc_type<std::complex<Tfloat>>(osize);

    std::complex<Tfloat>* gpu_in     = NULL;
    hipError_t            hip_status = hipSuccess;
    hip_status                       = hipMalloc(&gpu_in, isize * sizeof(std::complex<Tfloat>));
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    std::complex<Tfloat>* gpu_out = inplace ? gpu_in : NULL;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, osize * sizeof(std::complex<Tfloat>));
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    }

    // Set up the CPU plan:
    auto cpu_plan = fftw_plan_guru64_dft<Tfloat>(dims.size(),
                                                 dims.data(),
                                                 howmany_dims.size(),
                                                 howmany_dims.data(),
                                                 reinterpret_cast<fftw_complex_type*>(cpu_in),
                                                 reinterpret_cast<fftw_complex_type*>(cpu_out),
                                                 sign,
                                                 FFTW_ESTIMATE);
    ASSERT_TRUE(cpu_plan != NULL) << "FFTW plan creation failure";

    // Set up the GPU plan:
    rocfft_status fft_status = rocfft_status_success;

    rocfft_plan_description gpu_description = NULL;
    fft_status                              = rocfft_plan_description_create(&gpu_description);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";

    std::vector<size_t> istride
        = {static_cast<size_t>(dims[1].is), static_cast<size_t>(dims[0].is)};
    std::vector<size_t> ostride
        = {static_cast<size_t>(dims[1].os), static_cast<size_t>(dims[0].os)};

    fft_status = rocfft_plan_description_set_data_layout(gpu_description,
                                                         rocfft_array_type_complex_interleaved,
                                                         rocfft_array_type_complex_interleaved,
                                                         NULL,
                                                         NULL,
                                                         istride.size(),
                                                         istride.data(),
                                                         howmany_dims[0].is,
                                                         ostride.size(),
                                                         ostride.data(),
                                                         howmany_dims[0].os);
    ASSERT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    rocfft_plan gpu_plan = NULL;
    fft_status
        = rocfft_plan_create(&gpu_plan,
                             inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                             transform_type,
                             precision_selector<Tfloat>(),
                             length.size(), // Dimension
                             length.data(), // lengths
                             batch, // Number of transforms
                             gpu_description); // Description
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info planinfo = NULL;
    fft_status                     = rocfft_execution_info_create(&planinfo);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";
    void* wbuffer = NULL;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
        fft_status = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }

    // Set up the data:
    std::fill(cpu_in, cpu_in + isize, 0.0);
    srandom(3);
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Ny; j++)
            {
                // TODO: make pattern variable?
                std::complex<Tfloat> val((Tfloat)rand() / (Tfloat)RAND_MAX,
                                         (Tfloat)rand() / (Tfloat)RAND_MAX);
                cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j] = val;
            }
        }
    }

    if(verbose > 1)
    {
        std::cout << "input:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Ny; j++)
                {
                    std::cout
                        << cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat input:\n";
        for(size_t i = 0; i < isize; ++i)
        {
            std::cout << cpu_in[i] << " ";
        }
        std::cout << std::endl;
    }

    hip_status
        = hipMemcpy(gpu_in, cpu_in, isize * sizeof(std::complex<Tfloat>), hipMemcpyHostToDevice);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    // Execute the GPU transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)&gpu_in, // in_buffer
                                (void**)&gpu_out, // out_buffer
                                planinfo); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Execute the CPU transform:
    fftw_execute_type<Tfloat>(cpu_plan);

    if(verbose > 1)
    {
        std::cout << "cpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Ny; j++)
                {
                    std::cout
                        << cpu_out[howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat cpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << cpu_out[i] << " ";
        }
        std::cout << std::endl;
    }

    // Copy the data back and compare:
    fftw_vector<std::complex<Tfloat>> gpu_out_comp(osize);
    hip_status = hipMemcpy(
        gpu_out_comp.data(), gpu_out, osize * sizeof(std::complex<Tfloat>), hipMemcpyDeviceToHost);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    if(verbose > 1)
    {
        std::cout << "gpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Ny; j++)
                {
                    std::cout << gpu_out_comp[howmany_dims[0].os * ibatch + dims[0].os * i
                                              + dims[1].os * j]
                              << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat gpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << gpu_out_comp[i] << " ";
        }
        std::cout << std::endl;
    }

    Tfloat L2norm   = 0.0;
    Tfloat Linfnorm = 0.0;
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Ny; j++)
            {
                auto val = cpu_out[howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j];
                Linfnorm = std::max(std::abs(val), Linfnorm);
                L2norm += std::abs(val) * std::abs(val);
            }
        }
    }
    L2norm = sqrt(L2norm);

    Tfloat L2diff   = 0.0;
    Tfloat Linfdiff = 0.0;
    int    nwrong   = 0;
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Ny; j++)
            {
                const int pos  = howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j;
                Tfloat    diff = std::abs(cpu_out[pos] - gpu_out_comp[pos]);
                Linfdiff       = std::max(diff, Linfdiff);
                L2diff += diff * diff;
                if(std::abs(diff) / (Linfnorm * log(Nx * Ny)) >= type_epsilon<Tfloat>())
                {
                    nwrong++;
                    if(verbose > 1)
                    {
                        std::cout << "ibatch: " << ibatch << ", (i,j): " << i << " " << j
                                  << ", cpu: " << cpu_out[pos] << ", gpu: " << gpu_out_comp[pos]
                                  << "\n";
                    }
                }
            }
        }
    }
    L2diff = sqrt(L2diff);

    Tfloat L2error   = L2diff / (L2norm * sqrt(log(Nx * Ny)));
    Tfloat Linferror = Linfdiff / (Linfnorm * log(Nx * Ny));
    if(verbose > 1)
    {
        std::cout << "nwrong: " << nwrong << std::endl;
        std::cout << "relative L2 error: " << L2error << std::endl;
        std::cout << "relative Linf error: " << Linferror << std::endl;
    }

    EXPECT_TRUE(L2error < type_epsilon<Tfloat>())
        << "Tolerance failure: L2error: " << L2error << ", tolerance: " << type_epsilon<Tfloat>();
    EXPECT_TRUE(Linferror < type_epsilon<Tfloat>()) << "Tolerance failure: Linferror: " << Linferror
                                                    << ", tolerance: " << type_epsilon<Tfloat>();

    // Free GPU memory:
    hipFree(gpu_in);
    fftw_free(cpu_in);
    if(!inplace)
    {
        hipFree(gpu_out);
        fftw_free(cpu_out);
    }
    if(wbuffer != NULL)
    {
        hipFree(wbuffer);
    }

    // Destroy plans:
    rocfft_execution_info_destroy(planinfo);
    rocfft_plan_description_destroy(gpu_description);
    rocfft_plan_destroy(gpu_plan);
    fftw_destroy_plan_type(cpu_plan);
}

// Implemetation of complex-to-complex tests for float and double:

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_single_precision)
{
    size_t                  Nx             = std::get<0>(GetParam());
    size_t                  Ny             = std::get<1>(GetParam());
    size_t                  batch          = std::get<2>(GetParam());
    rocfft_result_placement placeness      = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());
    rocfft_transform_type   transform_type = std::get<6>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<float>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_double_precision)
{

    size_t                  Nx             = std::get<0>(GetParam());
    size_t                  Ny             = std::get<1>(GetParam());
    size_t                  batch          = std::get<2>(GetParam());
    rocfft_result_placement placeness      = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());
    rocfft_transform_type   transform_type = std::get<6>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<double>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Templated test function for real to complex:
template <typename Tfloat>
void normal_2D_real_to_complex_interleaved(size_t                  Nx,
                                           size_t                  Ny,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;

    std::vector<size_t> length = {Ny, Nx};

    const bool inplace = placeness == rocfft_placement_inplace;

    // TODO: add logic to deal with discontiguous data in Nystride
    const size_t Nycomplex = Ny / 2 + 1;
    const size_t Nystride  = inplace ? 2 * Nycomplex : Ny;

    // Dimension configuration:
    std::array<fftw_iodim64, 2> dims;
    dims[1].n  = Ny;
    dims[1].is = stride;
    dims[1].os = stride;
    dims[0].n  = Nx;
    dims[0].is = Nystride * dims[1].is;
    dims[0].os = (dims[1].n / 2 + 1) * dims[1].os;

    // Batch configuration:
    std::array<fftw_iodim64, 1> howmany_dims;
    howmany_dims[0].n  = batch;
    howmany_dims[0].is = dims[0].n * dims[0].is;
    howmany_dims[0].os = dims[0].n * dims[0].os;

    const size_t isize = howmany_dims[0].n * howmany_dims[0].is;
    const size_t osize = howmany_dims[0].n * howmany_dims[0].os;

    if(verbose)
    {
        if(inplace)
            std::cout << "in-place\n";
        else
            std::cout << "out-of-place\n";
        for(int i = 0; i < dims.size(); ++i)
        {
            std::cout << "dim " << i << std::endl;
            std::cout << "\tn: " << dims[i].n << std::endl;
            std::cout << "\tis: " << dims[i].is << std::endl;
            std::cout << "\tos: " << dims[i].os << std::endl;
        }
        std::cout << "howmany_dims:\n";
        std::cout << "\tn: " << howmany_dims[0].n << std::endl;
        std::cout << "\tis: " << howmany_dims[0].is << std::endl;
        std::cout << "\tos: " << howmany_dims[0].os << std::endl;
        std::cout << "isize: " << isize << "\n";
        std::cout << "osize: " << osize << "\n";
    }

    // Set up buffers:
    // Local data buffer:
    Tfloat* cpu_in = fftw_alloc_type<Tfloat>(isize);
    // Output buffer
    std::complex<Tfloat>* cpu_out
        = inplace ? (std::complex<Tfloat>*)cpu_in : fftw_alloc_type<std::complex<Tfloat>>(osize);

    Tfloat*    gpu_in     = NULL;
    hipError_t hip_status = hipSuccess;
    hip_status            = hipMalloc(&gpu_in, isize * sizeof(Tfloat));
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    std::complex<Tfloat>* gpu_out = inplace ? (std::complex<Tfloat>*)gpu_in : NULL;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, osize * sizeof(std::complex<Tfloat>));
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    }

    // Set up the CPU plan:
    auto cpu_plan = fftw_plan_guru64_r2c<Tfloat>(dims.size(),
                                                 dims.data(),
                                                 howmany_dims.size(),
                                                 howmany_dims.data(),
                                                 cpu_in,
                                                 reinterpret_cast<fftw_complex_type*>(cpu_out),
                                                 FFTW_ESTIMATE);
    ASSERT_TRUE(cpu_plan != NULL) << "FFTW plan creation failure";

    // Set up the GPU plan:
    rocfft_status fft_status = rocfft_status_success;

    rocfft_plan_description gpu_description = NULL;
    fft_status                              = rocfft_plan_description_create(&gpu_description);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";

    std::vector<size_t> istride
        = {static_cast<size_t>(dims[1].is), static_cast<size_t>(dims[0].is)};
    std::vector<size_t> ostride
        = {static_cast<size_t>(dims[1].os), static_cast<size_t>(dims[0].os)};

    fft_status = rocfft_plan_description_set_data_layout(gpu_description,
                                                         rocfft_array_type_real,
                                                         rocfft_array_type_hermitian_interleaved,
                                                         NULL,
                                                         NULL,
                                                         istride.size(),
                                                         istride.data(),
                                                         howmany_dims[0].is,
                                                         ostride.size(),
                                                         ostride.data(),
                                                         howmany_dims[0].os);
    ASSERT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    rocfft_plan gpu_plan = NULL;
    fft_status
        = rocfft_plan_create(&gpu_plan,
                             inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                             rocfft_transform_type_real_forward,
                             precision_selector<Tfloat>(),
                             length.size(), // Dimensions
                             length.data(), // lengths
                             batch, // Number of transforms
                             gpu_description); // Description
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info planinfo = NULL;
    fft_status                     = rocfft_execution_info_create(&planinfo);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";
    void* wbuffer = NULL;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
        fft_status = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }

    // Set up the data:
    std::fill(cpu_in, cpu_in + isize, 0.0);
    srandom(3);
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Ny; j++)
            {
                // TODO: make pattern variable
                cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j]
                    = (Tfloat)rand() / (Tfloat)RAND_MAX;
            }
        }
    }

    if(verbose > 1)
    {
        std::cout << "input:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Ny; j++)
                {
                    std::cout
                        << cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat input:\n";
        for(size_t i = 0; i < isize; ++i)
        {
            std::cout << cpu_in[i] << " ";
        }
        std::cout << std::endl;
    }

    hip_status = hipMemcpy(gpu_in, cpu_in, isize * sizeof(Tfloat), hipMemcpyHostToDevice);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    // Execute the GPU transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)&gpu_in, // in_buffer
                                (void**)&gpu_out, // out_buffer
                                planinfo); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Execute the CPU transform:
    fftw_execute_type<Tfloat>(cpu_plan);

    if(verbose > 1)
    {
        std::cout << "cpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Nycomplex; j++)
                {
                    std::cout
                        << cpu_out[howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat cpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << cpu_out[i] << " ";
        }
        std::cout << std::endl;
    }

    // Copy the data back and compare:
    fftw_vector<std::complex<Tfloat>> gpu_out_comp(osize);
    hip_status = hipMemcpy(
        gpu_out_comp.data(), gpu_out, osize * sizeof(std::complex<Tfloat>), hipMemcpyDeviceToHost);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    if(verbose > 1)
    {
        std::cout << "gpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Nycomplex; j++)
                {
                    std::cout << gpu_out_comp[howmany_dims[0].os * ibatch + dims[0].os * i
                                              + dims[1].os * j]
                              << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat gpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << gpu_out_comp[i] << " ";
        }
        std::cout << std::endl;
    }

    Tfloat L2diff   = 0.0;
    Tfloat Linfdiff = 0.0;
    Tfloat L2norm   = 0.0;
    Tfloat Linfnorm = 0.0;
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Nycomplex; j++)
            {
                const int pos  = howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j;
                Tfloat    diff = std::abs(cpu_out[pos] - gpu_out_comp[pos]);
                Linfnorm       = std::max(std::abs(cpu_out[pos]), Linfnorm);
                L2norm += std::abs(cpu_out[pos]) * std::abs(cpu_out[pos]);
                Linfdiff = std::max(diff, Linfdiff);
                L2diff += diff * diff;
            }
        }
    }
    L2norm = sqrt(L2norm);
    L2diff = sqrt(L2diff);

    Tfloat L2error   = L2diff / (L2norm * sqrt(log(Nx * Ny)));
    Tfloat Linferror = Linfdiff / (Linfnorm * log(Nx * Ny));
    if(verbose)
    {
        std::cout << "relative L2 error: " << L2error << std::endl;
        std::cout << "relative Linf error: " << Linferror << std::endl;
    }

    EXPECT_TRUE(L2error < type_epsilon<Tfloat>())
        << "Tolerance failure: L2error: " << L2error << ", tolerance: " << type_epsilon<Tfloat>();
    EXPECT_TRUE(Linferror < type_epsilon<Tfloat>()) << "Tolerance failure: Linferror: " << Linferror
                                                    << ", tolerance: " << type_epsilon<Tfloat>();

    // Free GPU memory:
    hipFree(gpu_in);
    fftw_free(cpu_in);
    if(!inplace)
    {
        hipFree(gpu_out);
        fftw_free(cpu_out);
    }
    if(wbuffer != NULL)
    {
        hipFree(wbuffer);
    }

    // Destroy plans:
    rocfft_execution_info_destroy(planinfo);
    rocfft_plan_description_destroy(gpu_description);
    rocfft_plan_destroy(gpu_plan);
    fftw_destroy_plan_type(cpu_plan);
}

// Impose Hermitian symmetry on a 3D complex array of size Nx * (Ny / 2 + 1) with batches.
template <typename Tfloat>
void imposeHermitianSymmetry(std::complex<Tfloat>*              data,
                             const std::array<fftw_iodim64, 2>& dims,
                             const fftw_iodim64                 howmany_dims)
{
    auto Nx = dims[0].n;
    auto Ny = dims[1].n;

    for(size_t ibatch = 0; ibatch < howmany_dims.n; ++ibatch)
    {
        // origin:
        data[howmany_dims.is * ibatch + 0].imag(0.0);

        if(Nx % 2 == 0)
        {
            // x-Nyquist is real-valued
            data[howmany_dims.is * ibatch + dims[0].is * (Nx / 2)].imag(0.0);
        }
        if(Ny % 2 == 0)
        {
            // y-Nyquist is real-valued
            data[howmany_dims.is * ibatch + dims[1].is * (Ny / 2)].imag(0.0);
        }
        if(Nx % 2 == 0 && Ny % 2 == 0)
        {
            // xy-Nyquist is real-valued
            data[howmany_dims.is * ibatch + dims[0].is * (Nx / 2) + dims[1].is * (Ny / 2)].imag(
                0.0);
        }

        // x-axis:
        for(int i = 1; i < Nx / 2 + 1; ++i)
        {
            data[howmany_dims.is * ibatch + dims[0].is * (Nx - i)]
                = std::conj(data[howmany_dims.is * ibatch + dims[0].is * i]);
        }

        // y-Nyquist:
        if(Ny % 2 == 0)
        {
            for(int i = 1; i < Nx / 2 + 1; ++i)
            {
                data[howmany_dims.is * ibatch + dims[0].is * (Nx - i) + dims[1].is * (Ny / 2)]
                    = std::conj(
                        data[howmany_dims.is * ibatch + dims[0].is * i + dims[1].is * (Ny / 2)]);
            }
        }
    }
}

// Check for exact Hermitian symmetry on a 2D complex array of size Nx  * (Ny / 2 + 1)
// with multiple batches.
template <typename Tfloat>
bool isHermitianSymmetric(std::complex<Tfloat>*              data,
                          const std::array<fftw_iodim64, 2>& dims,
                          const fftw_iodim64                 howmany_dims)
{
    for(size_t ibatch = 0; ibatch < howmany_dims.n; ++ibatch)
    {
        for(int i = 0; i < dims[0].n; ++i)
        {
            int i0 = (dims[0].n - i) % dims[0].n;
            for(int j = 0; j < dims[1].n / 2 + 1; ++j)
            {
                int j0 = (dims[1].n - j) % dims[1].n;
                if(j0 < dims[1].n / 2 + 1)
                {
                    auto pos  = howmany_dims.is * ibatch + dims[0].is * i + dims[1].is * j;
                    auto pos0 = howmany_dims.is * ibatch + dims[0].is * i0 + dims[1].is * j0;
                    if(data[pos] != std::conj(data[pos0]))
                    {
                        if(verbose)
                        {
                            std::cout << "ibatch: " << ibatch << std::endl;
                            std::cout << "i,j:    " << i << "," << j << " -> " << pos << " -> "
                                      << data[pos] << std::endl;
                            std::cout << "i0,j0:  " << i0 << "," << j0 << " -> " << pos0 << " -> "
                                      << data[pos0] << std::endl;
                        }
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// Templated test function for real to complex:
template <typename Tfloat>
void normal_2D_complex_interleaved_to_real(size_t                  Nx,
                                           size_t                  Ny,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;

    std::vector<size_t> length  = {Ny, Nx};
    const bool          inplace = placeness == rocfft_placement_inplace;

    // TODO: add logic to deal with discontiguous data in Nystride
    const size_t Nycomplex = Ny / 2 + 1;
    const size_t Nystride  = inplace ? 2 * Nycomplex : Ny;

    // Dimension configuration:
    std::array<fftw_iodim64, 2> dims;
    dims[1].n  = Ny;
    dims[1].is = stride;
    dims[1].os = stride;
    dims[0].n  = Nx;
    dims[0].is = (dims[1].n / 2 + 1) * dims[1].is;
    dims[0].os = Nystride * dims[1].os;

    // Batch configuration:
    std::array<fftw_iodim64, 1> howmany_dims;
    howmany_dims[0].n  = batch;
    howmany_dims[0].is = dims[0].n * dims[0].is;
    howmany_dims[0].os = dims[0].n * dims[0].os;

    const size_t isize = howmany_dims[0].n * howmany_dims[0].is;
    const size_t osize = howmany_dims[0].n * howmany_dims[0].os;

    if(verbose)
    {
        if(inplace)
            std::cout << "in-place\n";
        else
            std::cout << "out-of-place\n";
        for(int i = 0; i < dims.size(); ++i)
        {
            std::cout << "dim " << i << std::endl;
            std::cout << "\tn: " << dims[i].n << std::endl;
            std::cout << "\tis: " << dims[i].is << std::endl;
            std::cout << "\tos: " << dims[i].os << std::endl;
        }
        std::cout << "howmany_dims:\n";
        std::cout << "\tn: " << howmany_dims[0].n << std::endl;
        std::cout << "\tis: " << howmany_dims[0].is << std::endl;
        std::cout << "\tos: " << howmany_dims[0].os << std::endl;
        std::cout << "isize: " << isize << "\n";
        std::cout << "osize: " << osize << "\n";
    }

    // Set up buffers:
    // Local data buffer:
    std::complex<Tfloat>* cpu_in = fftw_alloc_type<std::complex<Tfloat>>(isize);
    // Output buffer
    Tfloat* cpu_out = inplace ? (Tfloat*)cpu_in : fftw_alloc_type<Tfloat>(osize);

    std::complex<Tfloat>* gpu_in     = NULL;
    hipError_t            hip_status = hipSuccess;
    hip_status                       = hipMalloc(&gpu_in, isize * sizeof(std::complex<Tfloat>));
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    Tfloat* gpu_out = inplace ? (Tfloat*)gpu_in : NULL;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, osize * sizeof(Tfloat));
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    }

    // Set up the CPU plan:
    auto cpu_plan = fftw_plan_guru64_c2r<Tfloat>(dims.size(),
                                                 dims.data(),
                                                 howmany_dims.size(),
                                                 howmany_dims.data(),
                                                 reinterpret_cast<fftw_complex_type*>(cpu_in),
                                                 cpu_out,
                                                 FFTW_ESTIMATE);
    ASSERT_TRUE(cpu_plan != NULL) << "FFTW plan creation failure";

    // Set up the GPU plan:
    rocfft_status           fft_status      = rocfft_status_success;
    rocfft_plan_description gpu_description = NULL;
    fft_status                              = rocfft_plan_description_create(&gpu_description);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";

    std::vector<size_t> istride
        = {static_cast<size_t>(dims[1].is), static_cast<size_t>(dims[0].is)};
    std::vector<size_t> ostride
        = {static_cast<size_t>(dims[1].os), static_cast<size_t>(dims[0].os)};

    fft_status = rocfft_plan_description_set_data_layout(gpu_description,
                                                         rocfft_array_type_hermitian_interleaved,
                                                         rocfft_array_type_real,
                                                         NULL,
                                                         NULL,
                                                         istride.size(),
                                                         istride.data(),
                                                         howmany_dims[0].is,
                                                         ostride.size(),
                                                         ostride.data(),
                                                         howmany_dims[0].os);
    ASSERT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    rocfft_plan gpu_plan = NULL;
    fft_status
        = rocfft_plan_create(&gpu_plan,
                             inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                             rocfft_transform_type_real_inverse,
                             precision_selector<Tfloat>(),
                             length.size(), // Dimensions
                             length.data(), // lengths
                             batch, // Number of transforms
                             gpu_description);
    ASSERT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT plan creation failure: " << fft_status;

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info planinfo = NULL;
    fft_status                     = rocfft_execution_info_create(&planinfo);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";
    void* wbuffer = NULL;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
        fft_status = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }

    // Set up the data:
    std::fill(cpu_in, cpu_in + isize, 0.0);
    srandom(3);
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < dims[0].n; i++)
        {
            for(size_t j = 0; j < dims[1].n / 2 + 1; j++)
            {
                // TODO: make pattern variable
                cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j]
                    = std::complex<Tfloat>((Tfloat)rand() / (Tfloat)RAND_MAX,
                                           (Tfloat)rand() / (Tfloat)RAND_MAX);
            }
        }
    }

    imposeHermitianSymmetry(cpu_in, dims, howmany_dims[0]);

    if(verbose > 1)
    {
        std::cout << "\ninput:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < Nx; i++)
            {
                for(size_t j = 0; j < Nycomplex; j++)
                {
                    std::cout
                        << cpu_in[howmany_dims[0].is * ibatch + dims[0].is * i + dims[1].is * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat input:\n";
        for(size_t i = 0; i < isize; ++i)
        {
            std::cout << cpu_in[i] << " ";
        }
        std::cout << std::endl;
    }

    ASSERT_TRUE(isHermitianSymmetric(cpu_in, dims, howmany_dims[0]));

    hip_status
        = hipMemcpy(gpu_in, cpu_in, isize * sizeof(std::complex<Tfloat>), hipMemcpyHostToDevice);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    // Execute the GPU transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)&gpu_in, // in_buffer
                                (void**)&gpu_out, // out_buffer
                                planinfo); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Execute the CPU transform:
    fftw_execute_type<Tfloat>(cpu_plan);

    if(verbose > 1)
    {
        std::cout << "cpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < dims[0].n; i++)
            {
                for(size_t j = 0; j < dims[1].n; j++)
                {
                    std::cout
                        << cpu_out[howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j]
                        << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat cpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << cpu_out[i] << " ";
        }
        std::cout << std::endl;
    }

    // Copy the data back and compare:
    std::vector<Tfloat> gpu_out_comp(osize);
    hip_status
        = hipMemcpy(gpu_out_comp.data(), gpu_out, osize * sizeof(Tfloat), hipMemcpyDeviceToHost);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    if(verbose > 1)
    {
        std::cout << "gpu_out:\n";
        for(size_t ibatch = 0; ibatch < batch; ++ibatch)
        {
            std::cout << "ibatch: " << ibatch << std::endl;
            for(size_t i = 0; i < dims[0].n; i++)
            {
                for(size_t j = 0; j < dims[1].n; j++)
                {
                    std::cout << gpu_out_comp[howmany_dims[0].os * ibatch + dims[0].os * i
                                              + dims[1].os * j]
                              << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    if(verbose > 2)
    {
        std::cout << "flat gpu output:\n";
        for(size_t i = 0; i < osize; ++i)
        {
            std::cout << gpu_out_comp[i] << " ";
        }
        std::cout << std::endl;
    }

    Tfloat L2diff   = 0.0;
    Tfloat Linfdiff = 0.0;
    Tfloat L2norm   = 0.0;
    Tfloat Linfnorm = 0.0;
    for(size_t ibatch = 0; ibatch < batch; ++ibatch)
    {
        for(size_t i = 0; i < dims[0].n; i++)
        {
            for(size_t j = 0; j < dims[1].n; j++)
            {
                const int pos  = howmany_dims[0].os * ibatch + dims[0].os * i + dims[1].os * j;
                Tfloat    diff = std::abs(cpu_out[pos] - gpu_out_comp[pos]);
                Linfnorm       = std::max(std::abs(cpu_out[pos]), Linfnorm);
                L2norm += std::abs(cpu_out[pos]) * std::abs(cpu_out[pos]);
                Linfdiff = std::max(diff, Linfdiff);
                L2diff += diff * diff;
            }
        }
    }
    L2norm = sqrt(L2norm);
    L2diff = sqrt(L2diff);

    Tfloat L2error   = L2diff / (L2norm * sqrt(log(Nx * Ny)));
    Tfloat Linferror = Linfdiff / (Linfnorm * log(Nx * Ny));
    if(verbose)
    {
        std::cout << "relative L2 error: " << L2error << std::endl;
        std::cout << "relative Linf error: " << Linferror << std::endl;
    }

    EXPECT_TRUE(L2error < type_epsilon<Tfloat>())
        << "Tolerance failure: L2error: " << L2error << ", tolerance: " << type_epsilon<Tfloat>();
    EXPECT_TRUE(Linferror < type_epsilon<Tfloat>()) << "Tolerance failure: Linferror: " << Linferror
                                                    << ", tolerance: " << type_epsilon<Tfloat>();

    // Free GPU memory:
    hipFree(gpu_in);
    fftw_free(cpu_in);
    if(!inplace)
    {
        hipFree(gpu_out);
        fftw_free(cpu_out);
    }
    if(wbuffer != NULL)
    {
        hipFree(wbuffer);
    }

    // Destroy plans:
    rocfft_execution_info_destroy(planinfo);
    rocfft_plan_description_destroy(gpu_description);
    rocfft_plan_destroy(gpu_plan);
    fftw_destroy_plan_type(cpu_plan);
}

// Implemetation of real-to-complex tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_single_precision)
{
    size_t                  Nx             = std::get<0>(GetParam());
    size_t                  Ny             = std::get<1>(GetParam());
    size_t                  batch          = std::get<2>(GetParam());
    rocfft_result_placement placeness      = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());
    rocfft_transform_type   transform_type = rocfft_transform_type_real_forward;

    try
    {
        normal_2D_real_to_complex_interleaved<float>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_double_precision)
{
    size_t                  Nx             = std::get<0>(GetParam());
    size_t                  Ny             = std::get<1>(GetParam());
    size_t                  batch          = std::get<2>(GetParam());
    rocfft_result_placement placeness      = std::get<3>(GetParam());
    size_t                  stride         = std::get<4>(GetParam());
    data_pattern            pattern        = std::get<5>(GetParam());
    rocfft_transform_type   transform_type = rocfft_transform_type_real_forward;

    try
    {
        normal_2D_real_to_complex_interleaved<double>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Implemetation of complex-to-real tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_single_precision)
{
    size_t                  Nx        = std::get<0>(GetParam());
    size_t                  Ny        = std::get<1>(GetParam());
    size_t                  batch     = std::get<2>(GetParam());
    rocfft_result_placement placeness = std::get<3>(GetParam());
    size_t                  stride    = std::get<4>(GetParam());
    data_pattern            pattern   = std::get<5>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_2D_complex_interleaved_to_real<float>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_double_precision)
{
    size_t                  Nx        = std::get<0>(GetParam());
    size_t                  Ny        = std::get<1>(GetParam());
    size_t                  batch     = std::get<2>(GetParam());
    rocfft_result_placement placeness = std::get<3>(GetParam());
    size_t                  stride    = std::get<4>(GetParam());
    data_pattern            pattern   = std::get<5>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_2D_complex_interleaved_to_real<double>(
            Nx, Ny, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(prime_range),
                                           ValuesIn(prime_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

// Complex to real and real-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));
