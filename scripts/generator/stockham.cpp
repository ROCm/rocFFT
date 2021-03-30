#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

#include <fftw3.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include <rocrand/rocrand.hpp>

#include "timer.h"

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;

template <class T>
struct real_type;

template <>
struct real_type<hipComplex>
{
    typedef float type;
};

template <>
struct real_type<hipDoubleComplex>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

#include "butterfly_template.h"

//
// Random inputs
//
template <typename T>
vector<T> random_vector(size_t n)
{
    vector<T> x(n);

    rocrand_cpp::random_device                             rd;
    rocrand_cpp::mtgp32                                    engine(rd());
    rocrand_cpp::normal_distribution<float>                dist(0.0, 1.5);
    rocrand_cpp::uniform_real_distribution<real_type_t<T>> distribution;

    real_type_t<T>* rbuf;
    HIP_CHECK(hipMalloc(&rbuf, 2 * n * sizeof(real_type_t<T>)));
    distribution(engine, rbuf, 2 * n);
    HIP_CHECK(hipMemcpy(x.data(), rbuf, 2 * n * sizeof(real_type_t<T>), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(rbuf));

    return x;
}

//
// Copy helper
//
template <typename T>
vector<T> copy(vector<T> const& x)
{
    vector<T> z(x.size());
    for(size_t i = 0; i < x.size(); ++i)
    {
        z[i] = x[i];
    }
    return z;
}

float average(vector<float> x)
{
    return accumulate(x.cbegin(), x.cend(), 0.0) / x.size();
}

//
// FFTW backed FFT
//
pair<float, vector<hipDoubleComplex>>
    fft_fftw(vector<hipDoubleComplex> const& x, int nx, int nbatch)
{
    // std::cout << "Complex Double" << std::endl;
    auto z = copy(x);
    // clang-format off
    auto p = fftw_plan_many_dft(1, &nx, nbatch,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftw_execute(p);
    timer.toc();
    fftw_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

pair<float, vector<hipComplex>> fft_fftw(vector<hipComplex> const& x, int nx, int nbatch)
{
    // std::cout << "Complex Single" << std::endl;
    auto z = copy(x);
    // clang-format off
    auto p = fftwf_plan_many_dft(1, &nx, nbatch,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftwf_execute(p);
    timer.toc();
    fftwf_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

//
// Stockham
//
template <typename T>
__global__ void stockham_twiddles(int ntwiddles, T* twiddles, int nfactors, int* factors)
{
    int m = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(m >= ntwiddles)
        return;

    int n       = 0;
    int factor  = factors[0];
    int nroots  = factor;
    int nraccum = factor - 1;
    while(m > nraccum - 1)
    {
        factor = factors[++n];
        nroots *= factor;
        nraccum += (factor - 1) * (nroots / factor);
    }

    int m0 = nraccum - (nroots / factor) * (factor - 1);
    int j  = (m - m0) % (factor - 1) + 1;
    int k  = (m - m0) / (factor - 1);

    // always use double for sine cosine, cast to real type when saving to twiddles
    double cost, sint;
    sincospi(-2 * double(j) * k / nroots, &sint, &cost);
    twiddles[m].x = real_type_t<T>(cost);
    twiddles[m].y = real_type_t<T>(sint);
}

#define TWIDDLE_MUL_FWD(TWIDDLES, INDEX, REG)  \
    {                                          \
        T              W = TWIDDLES[INDEX];    \
        real_type_t<T> TR, TI;                 \
        TR    = (W.x * REG.x) - (W.y * REG.y); \
        TI    = (W.y * REG.x) + (W.x * REG.y); \
        REG.x = TR;                            \
        REG.y = TI;                            \
    }

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

#include "stockham_generated_kernel.h"

template <typename T>
tuple<float, float, vector<T>> fft_stockham_gpu(vector<T> const& x, int nx, int nbatch)
{
    vector<float> times;
    int           ntrials = 10;

    auto z = copy(x);

    T* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(T), hipMemcpyHostToDevice));

    T* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx - 1) * sizeof(T)));

    GPUTimer total;
    total.tic();
    vector<int> factors;

    factors = GENERATED_FACTORS;

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, factors.size() * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(d_factors, factors.data(), factors.size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors.size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    if(false)
    {
        vector<T> t(nx - 1);
        HIP_CHECK(hipMemcpy(t.data(), twiddles, t.size() * sizeof(T), hipMemcpyDeviceToHost));
        for(int i = 0; i < nx - 1; ++i)
            cout << i << " " << t[i].x << " " << t[i].y << endl;
    }

    size_t* d_kargs;
    size_t  kargs[3];
    HIP_CHECK(hipMalloc(&d_kargs, 3 * sizeof(size_t)));

    kargs[0] = nx; // passed to global function as "lengths[0]"
    kargs[1] = 1; // passed to global function as "stride[0]"
    kargs[2] = nx; // passed to global function as "stride[1]"
    HIP_CHECK(hipMemcpy(d_kargs, kargs, 3 * sizeof(size_t), hipMemcpyHostToDevice));

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();
        // 1,1 means unit-stride (TODO: arbitrary stride)
        GENERATED_KERNEL_LAUNCH(X, nbatch, twiddles, d_kargs, 1, 1);

        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(T), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(d_kargs));
    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X));

    return {average(times), total.elapsed(), move(z)};
}

//
// Relative difference
//
template <typename T>
double compare(vector<T> const& z1, vector<T> const& z2)
{
    double d = 0.0;
    double r = 0.0;
    for(size_t n = 0; n < z1.size(); ++n)
    {
        double dx = z1[n].x - z2[n].x;
        double dy = z1[n].y - z2[n].y;
        // if(dx * dx + dy * dy > 1.e-7)
        // {
        //     cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n].x << " " << z2[n].x << endl;
        // }
        d += dx * dx + dy * dy;
        r += z1[n].x * z1[n].x + z1[n].y * z1[n].y;
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
template <typename T>
void test1d(size_t n, size_t nbatch)
{
    double GiB = double(n * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    double GB  = double(n * nbatch * sizeof(T)) / 1000 / 1000 / 1000;

    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:   " << GiB << " GiB" << endl;
    cout << "1d input size:   " << GB << " GB" << endl;

    auto x = random_vector<T>(n * nbatch);

    auto [t1, z1] = fft_fftw(x, n, nbatch);
    cout << "FFTW time:       " << t1 << " ms" << endl;

    auto [t2, t2t, z2] = fft_stockham_gpu(x, n, nbatch);

    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel time: " << t2 << " ms (average)" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / t2 << " GiB/s one-way" << endl;
    cout << "GPU throughput:  " << GB * 1000 / t2 << " GB/s one-way" << endl;
    cout << "GPU throughput:  " << 2 * GiB * 1000 / t2 << " GiB/s two-way" << endl;
    cout << "GPU throughput:  " << 2 * GB * 1000 / t2 << " GB/s two-way" << endl;
    cout << "GFLOPS:          " << nbatch * 5 * n * log(n) / log(2.0) / (1e6 * t2) << endl;
    cout << "TFLOPS:          " << nbatch * 5 * n * log(n) / log(2.0) / (1e9 * t2) << endl;
}

int main(int argc, char* argv[])
{
    size_t length = 256;
    size_t nbatch = 1;
    size_t single = 1;
    if(argc > 1)
        length = stoi(argv[1]);
    if(argc > 2)
        nbatch = stoi(argv[2]);
    if(argc > 3)
        single = stoi(argv[3]);

    if(single)
        test1d<hipComplex>(length, nbatch);
    else
        test1d<hipDoubleComplex>(length, nbatch);
}
