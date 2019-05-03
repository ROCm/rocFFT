#include <iomanip>
#include <iostream>

#include <fftw3.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>

int main()
{
    ///------------------------------------------------------------------------
    /// fftwf_plan_many_dft_r2c vs hipfftPlanMany
    /// 2D R2C forward in-place

    float checksum[3] = {0, 0, 0}; // 0 for input, 1 for fftw cpu result, 2 for hipfft gpu result
    int   rank        = 2;
    int   n[2]        = {5, 6};
    int   howmany     = 3; // batch size

    unsigned flags                       = FFTW_ESTIMATE;
    int      n1_complex_elements         = n[1] / 2 + 1;
    int      n1_padding_real_elements    = n1_complex_elements * 2;
    int      total_padding_real_elements = n[0] * n1_padding_real_elements * howmany;
    int      total_bytes                 = sizeof(float) * total_padding_real_elements;

    int istride    = 1;
    int ostride    = 1;
    int inembed[2] = {n[0], n1_padding_real_elements};
    int onembed[2] = {n[0], n1_complex_elements};
    int idist      = n[0] * n1_padding_real_elements;
    int odist      = n[0] * n1_complex_elements;

    float*         cpu_in_out = new float[total_padding_real_elements];
    fftwf_complex* cpu_out    = (fftwf_complex*)fftw_malloc(total_bytes);

    const fftwf_plan fftwPlan = fftwf_plan_many_dft_r2c(rank,
                                                        n,
                                                        howmany,
                                                        cpu_in_out,
                                                        inembed,
                                                        istride,
                                                        idist,
                                                        (fftwf_complex*)cpu_in_out,
                                                        onembed,
                                                        ostride,
                                                        odist,
                                                        flags);

    for(int i = 0; i < total_padding_real_elements; i++)
    {
        cpu_in_out[i] = (i + 20) % 11;
        checksum[0] += cpu_in_out[i];
    }

    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::right);
    std::cout << "fftw inputs:---------------------------------------------\n";
    for(int i = 0; i < howmany; i++)
    {
        std::cout << "----------batch " << i << std::endl;
        for(int j = 0; j < n[0]; j++)
        {
            for(int k = 0; k < n1_padding_real_elements; k++)
            {
                int idx = i * n[0] * n1_padding_real_elements + j * n1_padding_real_elements + k;
                std::cout << "[" << j << ", " << k << "]: (" << std::setprecision(2) << std::setw(6)
                          << cpu_in_out[idx] << ") ";
                checksum[0] += cpu_in_out[idx];
            }
            std::cout << std::endl;
        }
    }

    fftwf_execute(fftwPlan);

    std::cout << "fftw outputs:--------------------------------------------\n";
    for(int i = 0; i < howmany; i++)
    {
        std::cout << "----------batch " << i << std::endl;
        for(int j = 0; j < n[0]; j++)
        {
            for(int k = 0; k < n1_padding_real_elements; k++)
            {
                int idx = i * n[0] * n1_padding_real_elements + j * n1_padding_real_elements + k;
                std::cout << "[" << j << ", " << k << "]: (" << std::setprecision(2) << std::setw(6)
                          << cpu_in_out[idx] << ") ";
                checksum[1] += cpu_in_out[idx];
            }
            std::cout << std::endl;
        }
    }

    // reinitialize
    for(int i = 0; i < total_padding_real_elements; i++)
    {
        cpu_in_out[i] = (i + 20) % 11;
    }

    hipfftHandle hipForwardPlan;
    hipfftResult result;

    result = hipfftPlanMany(&hipForwardPlan,
                            rank,
                            n,
                            inembed,
                            istride,
                            idist,
                            onembed,
                            ostride,
                            odist,
                            HIPFFT_R2C,
                            howmany);

    hipfftReal* gpu_in_out;
    hipMalloc((void**)&gpu_in_out, total_bytes);
    hipMemcpy(gpu_in_out, (void*)cpu_in_out, total_bytes, hipMemcpyHostToDevice);

    result = hipfftExecR2C(hipForwardPlan, gpu_in_out, (hipfftComplex*)gpu_in_out);

    hipMemcpy((void*)cpu_in_out, gpu_in_out, total_bytes, hipMemcpyDeviceToHost);

    std::cout << "hipfft outputs:------------------------------------------\n";
    for(int i = 0; i < howmany; i++)
    {
        std::cout << "----------batch " << i << std::endl;
        for(int j = 0; j < n[0]; j++)
        {
            for(int k = 0; k < n1_padding_real_elements; k++)
            {
                int idx = i * n[0] * n1_padding_real_elements + j * n1_padding_real_elements + k;
                std::cout << "[" << j << ", " << k << "]: (" << std::setprecision(2) << std::setw(6)
                          << cpu_in_out[idx] << ") ";
                checksum[2] += cpu_in_out[idx];
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n----------------------------------------------------\n"
              << "Final sum, input: " << checksum[0] << ", fftw result: " << checksum[1]
              << ", hipfft result: " << checksum[2] << std::endl;

    hipfftDestroy(hipForwardPlan);
    hipFree(gpu_in_out);

    fftwf_destroy_plan(fftwPlan);
    fftwf_free(cpu_in_out);
    return 0;
}
