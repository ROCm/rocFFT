#pragma once

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return;

#define HIP_CHECK0(r)   \
    if(r != hipSuccess) \
        return 0;

struct GPUTimer
{
    hipEvent_t start, stop;

    GPUTimer()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    ~GPUTimer()
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void tic()
    {
        HIP_CHECK(hipEventRecord(start, 0));
    }

    void toc()
    {
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
    }

    float elapsed()
    {
        float elapsed;
        HIP_CHECK0(hipEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

struct CPUTimer
{
    std::chrono::time_point<std::chrono::steady_clock> start, stop;

    void tic()
    {
        start = std::chrono::steady_clock::now();
    }

    void toc()
    {
        stop = std::chrono::steady_clock::now();
    }

    float elapsed()
    {
        std::chrono::duration<float, std::milli> diff = stop - start;
        return diff.count();
    }
};

template <class Enum>
struct ClockTimer
{
    clock_t start[Enum::SIZE], stop[Enum::SIZE], total[Enum::SIZE];
    bool    accumulate = false;

    __device__ __host__ ClockTimer()
    {
        for(int c = 0; c < Enum::SIZE; ++c)
            total[c] = 0;
    }

    __device__ void tic(Enum c)
    {
        start[c] = clock();
    }

    __device__ void toc(Enum c)
    {
        stop[c] = clock();
        if(accumulate)
            total[c] += stop[c] - start[c];
    }
};

#undef HIP_CHECK
