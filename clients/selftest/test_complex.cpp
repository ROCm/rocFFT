/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "./basis_vector.h"
#include "rocfft.h"
#include <gtest/gtest.h>
#include <stdlib.h>

typedef float  complex_single[2];
typedef double complex_double[2];

inline bool SupportedLength(size_t len)
{
    size_t p = len;
    while(!(p % 2))
        p /= 2;
    while(!(p % 3))
        p /= 3;
    while(!(p % 5))
        p /= 5;

    if(p == 1)
        return true;
    else
        return false;
}

#define TOLERANCE 0.0001
#define MAXVAL_TOLERANCE 0.000001
#define MAXVAL_RELATIVE_TOLERANCE 0.01

template <typename T, typename CT>
void ErrorCheck(size_t N, CT* ref, CT* tst, size_t size = 0)
{
    if(size == 0)
    {
        size = N;
    }

    T maxRefReal = 0, maxRefImag = 0;
    T maxTstReal = 0, maxTstImag = 0;

    size_t maxRefRealIdx = 0, maxRefImagIdx = 0;
    size_t maxTstRealIdx = 0, maxTstImagIdx = 0;

#if 0 
    std::cout << "lib output" << std::endl;
	for(size_t i=0; i<size; i++)
	{
            std::cout << tst[i][0] << ", " << tst[i][1] << std::endl;
    }
    std::cout << "ref output" << std::endl;
	for(size_t i=0; i<size; i++)
	{
            std::cout << N*ref[i][0] << ", " << N*ref[i][1] << std::endl;
    }
#endif

    for(size_t i = 0; i < size; i++)
    {
        T refReal = N * fabs(ref[i][0]);
        T refImag = N * fabs(ref[i][1]);
        T tstReal = fabs(tst[i][0]);
        T tstImag = fabs(tst[i][1]);

        bool newMaxRefR = refReal > maxRefReal;
        bool newMaxRefI = refImag > maxRefImag;
        bool newMaxTstR = tstReal > maxTstReal;
        bool newMaxTstI = tstImag > maxTstImag;

        maxRefReal = newMaxRefR ? refReal : maxRefReal;
        maxRefImag = newMaxRefI ? refImag : maxRefImag;

        maxRefRealIdx = newMaxRefR ? i : maxRefRealIdx;
        maxRefImagIdx = newMaxRefI ? i : maxRefImagIdx;

        maxTstReal = newMaxTstR ? refReal : maxTstReal;
        maxTstImag = newMaxTstI ? refImag : maxTstImag;

        maxTstRealIdx = newMaxTstR ? i : maxTstRealIdx;
        maxTstImagIdx = newMaxTstI ? i : maxTstImagIdx;
    }

    if(maxRefReal > 0 || maxRefImag > 0 || maxTstReal > 0 || maxTstImag > 0)
    {
        EXPECT_LE(fabs(maxRefReal - maxTstReal), fabs(maxRefReal * MAXVAL_TOLERANCE));
        EXPECT_LE(fabs(maxRefImag - maxTstImag), fabs(maxRefImag * MAXVAL_TOLERANCE));

        EXPECT_LE(fabs(N * ref[maxRefRealIdx][0] - tst[maxRefRealIdx][0]),
                  fabs(maxRefReal * MAXVAL_TOLERANCE));
        EXPECT_LE(fabs(N * ref[maxTstRealIdx][0] - tst[maxTstRealIdx][0]),
                  fabs(maxRefReal * MAXVAL_TOLERANCE));

        EXPECT_LE(fabs(N * ref[maxRefImagIdx][1] - tst[maxRefImagIdx][1]),
                  fabs(maxRefImag * MAXVAL_TOLERANCE));
        EXPECT_LE(fabs(N * ref[maxTstImagIdx][1] - tst[maxTstImagIdx][1]),
                  fabs(maxRefImag * MAXVAL_TOLERANCE));

        for(size_t i = 0; i < size; i++)
        {
            T refReal = N * fabs(ref[i][0]);
            T refImag = N * fabs(ref[i][1]);
            T tstReal = fabs(tst[i][0]);
            T tstImag = fabs(tst[i][1]);

            if(fabs(refReal - tstReal) > fabs(refReal * TOLERANCE))
            {
                // EXPECT_EQ(0, refReal);
                EXPECT_LE(fabs(tstReal), MAXVAL_RELATIVE_TOLERANCE * maxTstReal);
            }

            if(fabs(refImag - tstImag) > fabs(refImag * TOLERANCE))
            {
                // EXPECT_EQ(0, refImag);
                EXPECT_LE(fabs(tstImag), MAXVAL_RELATIVE_TOLERANCE * maxTstImag);
            }
        }
    }
    else
        EXPECT_TRUE(true);
}

template <typename T, typename CT>
void ErrorCheckReal(size_t N, T* ref, T* tst, size_t size)
{
    CT *ref_complex, *out_complex;
    ref_complex = (CT*)malloc(size * sizeof(CT));
    out_complex = (CT*)malloc(size * sizeof(CT));

    for(int i = 0; i < size; i++)
    {
        ref_complex[i][0] = ref[i];
        ref_complex[i][1] = 0;
        out_complex[i][0] = tst[i];
        out_complex[i][1] = 0;
    }

    ErrorCheck<T, CT>(N, ref_complex, out_complex, size);
    free(ref_complex);
    free(out_complex);
}

// Test cases

struct TrLen1D
{
    size_t N0;
};

struct TrLen2D
{
    size_t N0;
    size_t N1;
};

struct TrLen3D
{
    size_t N0;
    size_t N1;
    size_t N2;
};

const TrLen1D Test1DLengths[] = {
    {1},
    {2},
    {4},
    {8},
    {16},
    {32},
    {64},
    {128},
    {256},
    {512},
    {1024},
    {2048},
    {4096},
    {8192},
    {16384},
    {32768},
    {65536},
    {131072},
    {262144},
    {524288},
    {1048576},
    {2097152},
    {4194304},
    {8388608},
    {16777216},
    {33554432},
    // {67108864}, {134217728},

    {3},
    {9},
    {27},
    {81},
    {243},
    {729},
    {2187},
    {6561},
    {19683},
    {59049},
    {177147},
    {531441},
    {5},
    {25},
    {125},
    {625},
    {3125},
    {15625},
    {78125},
    {390625},

    {6144},
    {26244},
    {18},
    {5184},
    {768},
    {486},
    {34992},
    {13122},
    {108},
    {24},
    {1458},
    {576},
    {1536},
    {24576},
    {20736},
    {442368},
    {6},
    {39366},
    {46656},
    {10368},
    {72},
    {93312},
    {39366},
    {3888},
    {98304},
    {576},
    {62208},
    {497664},
    {24576},
    {209952},
    {6912},
    {393216},
    {559872},
    {419904},
    {46656},
    {165888},
    {12288},
    {69984},
    {36},
    {236196},

    {12500},
    {156250},
    {8000},
    {1280},
    {50},
    {50000},
    {128000},
    {1250},
    {4000},
    {100},
    {40},
    {80000},
    {256000},
    {20000},
    {1250},
    {1600},
    {163840},
    {655360},
    {250},
    {31250},
    {320},
    {80000},
    {125000},
    {31250},
    {625000},
    {2000},
    {1000000},
    {160},
    {1280},
    {40},
    {1000000},
    {1000000},
    {1250},
    {1600},
    {10240},
    {250000},
    {62500},
    {3200},
    {2500},
    {819200},

    {234375},
    {703125},
    {10125},
    {15},
    {6075},
    {15},
    {675},
    {15},
    {2025},
    {455625},
    {225},
    {9375},
    {98415},
    {421875},
    {375},
    {3645},
    {32805},
    {50625},
    {140625},
    {135},
    {140625},
    {151875},
    {32805},
    {10935},
    {18225},
    {18225},
    {405},
    {225},
    {140625},
    {455625},
    {151875},
    {820125},
    {151875},
    {75},
    {1875},
    {75},
    {820125},
    {164025},
    {5625},
    {10125},

    {648000},
    {3840},
    {12150},
    {2700},
    {120},
    {911250},
    {14400},
    {750},
    {233280},
    {829440},
    {562500},
    {607500},
    {7500},
    {360000},
    {92160},
    {22500},
    {97200},
    {36450},
    {21870},
    {729000},
    {6480},
    {900},
    {4500},
    {15000},
    {5760},
    {7680},
    {468750},
    {2250},
    {109350},
    {1620},
    {349920},
    {2400},
    {7680},
    {1920},
    {58320},
    {874800},
    {135000},
    {300000},
    {14400},
    {648000},
    {607500},
    {19200},
    {9000},
    {184320},
    {24300},
    {720},
    {328050},
    {30000},
    {607500},
    {590490},
    {233280},
    {75000},
    {524880},
    {32400},
    {38880},
    {48600},
    {240000},
    {225000},
    {691200},
    {777600},
    {150000},
    {12960},
    {51840},
    {30},
    {24000},
    {1200},
    {96000},
    {600000},
    {7500},
    {86400},
    {960000},
    {900000},
    {115200},
    {129600},
    {384000},
    {202500},
    {262440},
    {60},
    {468750},
    {103680},

    //{1594323}, {4782969},
    //{1953125}
};

const TrLen2D Test2DLengths[] = {
    {1, 1},      {2, 2},      {4, 4},      {8, 8},       {16, 16},    {32, 32},    {64, 64},
    {128, 128},  {256, 256},  {512, 512},  {1024, 1024}, {3, 3},      {9, 9},      {27, 27},
    {81, 81},    {243, 243},  {729, 729},  {5, 5},       {25, 25},    {125, 125},  {625, 625},

    {2700, 120}, {625, 288},  {90, 5},     {4, 10125},   {4, 10125},  {9, 1875},   {576, 576},
    {900, 270},  {15625, 25}, {100, 729},  {120, 729},   {4, 19440},  {100, 16},   {50, 3072},
    {10000, 24}, {10000, 24}, {27648, 36}, {6750, 128},  {38880, 24}, {15, 648},   {15360, 18},
    {15360, 18}, {768, 750},  {16, 720},   {16, 720},    {16, 720},   {135, 3456}, {80, 8748},
    {800, 450},  {3125, 32},  {1944, 320}, {1296, 486},  {81, 5400},  {81, 5400},  {81, 5400},
    {60, 30},    {1152, 25},  {1152, 25},  {3, 52488},   {1875, 108}, {1350, 512}, {256, 384},
    {27648, 5},  {27648, 5},  {72, 128},   {72, 128},    {125, 648},  {125, 648},  {150, 50},
    {150, 50},   {150, 50},   {216, 4000}, {24, 8192},   {24, 480},   {9, 73728},  {675, 1280},
    {675, 1280}, {21600, 40}, {21600, 40}, {81, 12000},  {13122, 10}, {486, 32},   {162, 216},
    {5400, 2},   {187500, 2}, {800, 400},  {8, 2592},    {5184, 30},  {10368, 15}, {160, 810},
    {2, 5120},   {7776, 75},  {6, 7290},   {48600, 2},   {360, 2048}, {20, 270},   {60, 1500},
    {60, 1350},  {1296, 25},  {486, 96},   {40, 1125},   {92160, 4},  {20, 2250},  {2, 233280},
    {10, 2500},  {5, 40000},  {2560, 72},  {2560, 72},   {2560, 72},  {30, 13500}, {30, 13500},
    {30, 2592},  {30, 2592},  {13500, 30}, {250, 300},   {250, 300},  {192, 648},  {4, 233280},
    {4, 233280}, {288, 1440}, {576, 1440}, {10368, 4},   {1440, 4},   {1440, 4},   {720, 90},
    {720, 90},   {720, 90},   {27, 18750}, {4000, 32},   {360, 32},   {10, 22500}, {10, 22500},
    {256, 3000}, {81, 12800}, {405, 1944}, {3, 2304},    {1728, 25},  {3888, 30},  {1536, 432},
    {225, 1200}, {405, 100},  {8000, 96},  {160, 45},    {24, 16000}, {1440, 300}, {58320, 15},
    {128, 2560}, {10, 486},   {15552, 2},  {192, 2000},  {25000, 20}, {40, 19200}, {40, 19200},
    {1000, 625}, {1200, 405}, {32, 120},   {15360, 10},  {15360, 10}, {4050, 8},   {4050, 8},
    {6, 100},    {625, 480},  {6075, 162}, {864, 972},   {384, 2500}, {144, 4320}, {40, 12150},
    {8, 288},    {8, 288},    {216, 3125}, {216, 3125},  {2700, 27},  {40000, 12}, {9, 23328},
    {480, 432},  {1500, 480}, {160, 1440}, {10125, 100}, {150, 250},  {120, 200},  {240, 2},
    {262440, 2}, {270, 2000}, {150, 1350}, {150, 1350},  {150, 810},  {18, 2430},  {60000, 6},
    {800, 768},  {324, 12},   {25, 432},   {25, 432},    {864, 192},  {4, 120000}, {360, 32},
    {32, 450},   {115200, 5}, {4500, 36},  {625, 60},    {625, 1458}, {432, 1600}, {10, 81920},
    {5400, 96},  {12500, 16}, {12500, 16}, {2430, 216},  {2430, 216}, {81000, 8},  {6, 36864},
    {19200, 10}, {12800, 45}, {15, 1000},  {216, 2880},  {25, 41472}, {1024, 180}, {1024, 180},
    {1024, 180}, {30720, 25}, {192, 972},  {192000, 5},

};

const TrLen3D Test3DLengths[] = {
    {1, 1, 1},      {15, 45, 1280}, {100, 8, 120},  {1600, 5, 108}, {15, 2, 100},   {25, 288, 128},
    {480, 375, 4},  {9, 375, 288},  {15, 48, 72},   {15, 48, 72},   {2250, 32, 3},  {4050, 25, 10},
    {20, 256, 50},  {24, 60, 15},   {24, 60, 15},   {60, 1152, 4},  {18, 10, 729},  {270, 45, 50},
    {250, 20, 90},  {10, 9, 8},     {150, 90, 5},   {1728, 5, 8},   {128, 9, 9},    {4, 72900, 3},
    {40500, 6, 2},  {40500, 6, 2},  {216, 24, 30},  {6, 3750, 24},  {3240, 4, 3},   {45, 24, 810},
    {243, 45, 54},  {400, 5, 144},  {4, 54, 200},   {324, 8, 20},   {3, 270, 320},  {3, 216, 8},
    {6, 96, 120},   {10, 4, 3072},  {16, 3645, 10}, {120, 144, 24}, {3, 3840, 54},  {12, 500, 120},
    {25, 16, 810},  {32, 32, 135},  {30, 36, 90},   {48, 30, 75},   {1000, 12, 64}, {243, 16, 5},
    {384, 15, 60},  {625, 18, 6},   {15, 324, 60},  {54, 27, 144},  {9, 8748, 9},   {32, 8, 2250},
    {2000, 10, 18}, {18, 5, 1024},  {750, 9, 36},   {9, 10, 6},     {540, 32, 48},  {6, 3, 40500},
    {300, 9, 225},  {45, 9, 48},    {3, 27, 450},   {2, 864, 162},  {30, 200, 25},  {9, 720, 100},
    {2, 24, 3456},  {2048, 27, 6},  {30, 64, 10},   {96, 72, 144},  {2, 125, 1500}, {216, 27, 135},
    {2, 360, 90},   {96, 162, 4},   {10, 6, 40},    {9, 24, 270},   {3, 450, 81},   {405, 32, 10},
    {15, 23328, 2}, {320, 10, 180}, {81, 2400, 2},  {768, 300, 3},  {6, 25, 64},    {10, 5, 50},
    {20, 25, 90},   {64, 15, 3},    {1152, 10, 25}, {12, 7200, 5},  {6, 150, 12},   {150, 12, 200},
    {5, 32, 1875},  {3200, 20, 6},  {2, 3125, 3},
};

template <typename T, typename CT>
class BasicInterfaceBasisTest : public ::testing::Test
{
protected:
    virtual void TearDown() {}
    virtual void SetUp()
    {
        in  = NULL;
        out = NULL;
        ref = NULL;
        dev = NULL;
        p   = NULL;
    }

    virtual void RunBvt(size_t L)
    {
        /*if(!SupportedLength(L))
    {
            return;
    }*/

        void* bufs[1];
        bufs[0] = dev;

        void*  workBuffer     = nullptr;
        size_t workBufferSize = 0;
        rocfft_plan_get_work_buffer_size(p, &workBufferSize);

        rocfft_execution_info info = nullptr;
        rocfft_execution_info_create(&info);

        if(workBufferSize > 0)
        {
            hipMalloc(&workBuffer, workBufferSize);
            rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize);
        }

        hipMemcpy(dev, in, L * sizeof(CT), hipMemcpyHostToDevice);
        rocfft_execute(p, bufs, NULL, info);
        hipDeviceSynchronize();
        hipMemcpy(out, dev, L * sizeof(CT), hipMemcpyDeviceToHost);

        ErrorCheck<T, CT>(L, ref, out);

        if(workBuffer)
            hipFree(workBuffer);

        rocfft_execution_info_destroy(info);
    }

    CT *        in, *out, *ref, *dev;
    rocfft_plan p;
};

template <typename T, typename CT, rocfft_precision prec>
class BasicInterface1DBasisTest : public BasicInterfaceBasisTest<T, CT>,
                                  public ::testing::WithParamInterface<TrLen1D>
{
protected:
    virtual void TestRoutine(size_t N, int dir)
    {
        size_t length[1];
        length[0] = N;

        FftBasisVectorMixComplex<T, CT> bvm(1, length, 1, FBC_RANDOM);
        bvm.RawPtrs(&this->in, &this->out, &this->dev);

        if(dir == -1)
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_forward,
                               prec,
                               1,
                               length,
                               1,
                               NULL);
        else
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_inverse,
                               prec,
                               1,
                               length,
                               1,
                               NULL);

        bvm.Generate(&this->in, &this->ref, dir);

        this->RunBvt(N);

        rocfft_plan_destroy(this->p);
    }
};

template <typename T, typename CT, rocfft_precision prec>
class BasicInterface2DBasisTest : public BasicInterfaceBasisTest<T, CT>,
                                  public ::testing::WithParamInterface<TrLen2D>
{
protected:
    virtual void TestRoutine(size_t N0, size_t N1, int dir)
    {
        size_t L = N0 * N1;
        size_t length[2];
        length[0] = N0;
        length[1] = N1;

        FftBasisVectorMixComplex<T, CT> bvm(2, length, 1, FBC_RANDOM);
        bvm.RawPtrs(&this->in, &this->out, &this->dev);

        if(dir == -1)
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_forward,
                               prec,
                               2,
                               length,
                               1,
                               NULL);
        else
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_inverse,
                               prec,
                               2,
                               length,
                               1,
                               NULL);

        bvm.Generate(&this->in, &this->ref, dir);

        this->RunBvt(L);

        rocfft_plan_destroy(this->p);
    }
};

template <typename T, typename CT, rocfft_precision prec>
class BasicInterface3DBasisTest : public BasicInterfaceBasisTest<T, CT>,
                                  public ::testing::WithParamInterface<TrLen3D>
{
protected:
    virtual void TestRoutine(size_t N0, size_t N1, size_t N2, int dir)
    {
        size_t L = N0 * N1 * N2;
        size_t length[3];
        length[0] = N0;
        length[1] = N1;
        length[2] = N2;

        FftBasisVectorMixComplex<T, CT> bvm(3, length, 1, FBC_RANDOM);
        bvm.RawPtrs(&this->in, &this->out, &this->dev);

        if(dir == -1)
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_forward,
                               prec,
                               3,
                               length,
                               1,
                               NULL);
        else
            rocfft_plan_create(&this->p,
                               rocfft_placement_inplace,
                               rocfft_transform_type_complex_inverse,
                               prec,
                               3,
                               length,
                               1,
                               NULL);

        bvm.Generate(&this->in, &this->ref, dir);

        this->RunBvt(L);

        rocfft_plan_destroy(this->p);
    }
};

// complex to complex cases
typedef BasicInterface1DBasisTest<double, complex_double, rocfft_precision_double>
    BasicInterfaceDouble1DBasisTest;
typedef BasicInterface2DBasisTest<double, complex_double, rocfft_precision_double>
    BasicInterfaceDouble2DBasisTest;
typedef BasicInterface3DBasisTest<double, complex_double, rocfft_precision_double>
    BasicInterfaceDouble3DBasisTest;

typedef BasicInterface1DBasisTest<float, complex_single, rocfft_precision_single>
    BasicInterfaceSingle1DBasisTest;
typedef BasicInterface2DBasisTest<float, complex_single, rocfft_precision_single>
    BasicInterfaceSingle2DBasisTest;
typedef BasicInterface3DBasisTest<float, complex_single, rocfft_precision_single>
    BasicInterfaceSingle3DBasisTest;

// complex to complex interface

// primes

TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinFwdLen504017)
{
    TestRoutine(504017, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinInvLen504017)
{
    TestRoutine(504017, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinFwdLen117191)
{
    TestRoutine(117191, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinInvLen117191)
{
    TestRoutine(117191, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinFwdLen7187)
{
    TestRoutine(7187, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinInvLen7187)
{
    TestRoutine(7187, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinFwdLen69)
{
    TestRoutine(69, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, BluesteinInvLen69)
{
    TestRoutine(69, 1);
}

TEST_F(BasicInterfaceSingle2DBasisTest, BluesteinFwdLen139and433)
{
    TestRoutine(139, 433, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, BluesteinInvLen139and433)
{
    TestRoutine(139, 433, 1);
}

TEST_F(BasicInterfaceSingle2DBasisTest, BluesteinFwdLen1061and229)
{
    TestRoutine(1061, 229, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, BluesteinInvLen1061and229)
{
    TestRoutine(1061, 229, 1);
}

TEST_F(BasicInterfaceSingle3DBasisTest, BluesteinFwdLen73and113and89)
{
    TestRoutine(73, 113, 89, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, BluesteinInvLen73and113and89)
{
    TestRoutine(73, 113, 89, 1);
}

TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinFwdLen504017)
{
    TestRoutine(504017, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinInvLen504017)
{
    TestRoutine(504017, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinFwdLen117191)
{
    TestRoutine(117191, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinInvLen117191)
{
    TestRoutine(117191, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinFwdLen7187)
{
    TestRoutine(7187, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinInvLen7187)
{
    TestRoutine(7187, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinFwdLen69)
{
    TestRoutine(69, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, BluesteinInvLen69)
{
    TestRoutine(69, 1);
}

TEST_F(BasicInterfaceDouble2DBasisTest, BluesteinFwdLen139and433)
{
    TestRoutine(139, 433, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, BluesteinInvLen139and433)
{
    TestRoutine(139, 433, 1);
}

TEST_F(BasicInterfaceDouble2DBasisTest, BluesteinFwdLen1061and229)
{
    TestRoutine(1061, 229, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, BluesteinInvLen1061and229)
{
    TestRoutine(1061, 229, 1);
}

TEST_F(BasicInterfaceDouble3DBasisTest, BluesteinFwdLen73and113and89)
{
    TestRoutine(73, 113, 89, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, BluesteinInvLen73and113and89)
{
    TestRoutine(73, 113, 89, 1);
}

// some big tests

/*
TEST_F( BasicInterfaceDouble1DBasisTest, FwdLen8388608 )		{
TestRoutine(8388608, -1); }
TEST_F( BasicInterfaceDouble1DBasisTest, FwdLen33554432 )		{
TestRoutine(33554432, -1); }
TEST_F( BasicInterfaceDouble1DBasisTest, InvLen8388608 )		{
TestRoutine(8388608, 1); }
TEST_F( BasicInterfaceDouble1DBasisTest, InvLen33554432 )		{
TestRoutine(33554432, 1); }

TEST_F( BasicInterfaceSingle1DBasisTest, FwdLen67108864 )		{
TestRoutine(67108864, -1); }
TEST_F( BasicInterfaceSingle1DBasisTest, FwdLen33554432 )		{
TestRoutine(33554432, -1); }
TEST_F( BasicInterfaceSingle1DBasisTest, InvLen67108864 )		{
TestRoutine(67108864, 1); }
TEST_F( BasicInterfaceSingle1DBasisTest, InvLen33554432 )		{
TestRoutine(33554432, 1); }
*/

// Basic Interface 1D tests
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen8)
{
    TestRoutine(8, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen7)
{
    TestRoutine(7, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen25)
{
    TestRoutine(25, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen81)
{
    TestRoutine(81, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen60)
{
    TestRoutine(60, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen64)
{
    TestRoutine(64, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen162)
{
    TestRoutine(162, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen100)
{
    TestRoutine(100, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen210)
{
    TestRoutine(210, -1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, FwdLen540_quick)
{
    TestRoutine(540, -1);
}

TEST_F(BasicInterfaceDouble1DBasisTest, InvLen8)
{
    TestRoutine(8, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen7)
{
    TestRoutine(7, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen25)
{
    TestRoutine(25, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen81)
{
    TestRoutine(81, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen60)
{
    TestRoutine(60, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen64)
{
    TestRoutine(64, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen162)
{
    TestRoutine(162, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen100)
{
    TestRoutine(100, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, InvLen210)
{
    TestRoutine(210, 1);
}
TEST_F(BasicInterfaceDouble1DBasisTest, invLen540_quick)
{
    TestRoutine(540, 1);
}

TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen390625)
{
    TestRoutine(390625, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen78125)
{
    TestRoutine(78125, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen1953125)
{
    TestRoutine(1953125, -1);
}
// TEST_F( BasicInterfaceSingle1DBasisTest, FwdLen48828125 )	{
// TestRoutine(48828125, -1); }

TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen14348907)
{
    TestRoutine(14348907, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen1594323)
{
    TestRoutine(1594323, -1);
}

TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen8)
{
    TestRoutine(8, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen7)
{
    TestRoutine(7, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen25)
{
    TestRoutine(25, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen81)
{
    TestRoutine(81, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen60)
{
    TestRoutine(60, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen64)
{
    TestRoutine(64, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen162)
{
    TestRoutine(162, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen100)
{
    TestRoutine(100, -1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, FwdLen210)
{
    TestRoutine(210, -1);
}

TEST_F(BasicInterfaceSingle1DBasisTest, InvLen8)
{
    TestRoutine(8, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen7)
{
    TestRoutine(7, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen25)
{
    TestRoutine(25, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen81)
{
    TestRoutine(81, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen60)
{
    TestRoutine(60, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen64)
{
    TestRoutine(64, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen162)
{
    TestRoutine(162, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen100)
{
    TestRoutine(100, 1);
}
TEST_F(BasicInterfaceSingle1DBasisTest, InvLen210)
{
    TestRoutine(210, 1);
}

// Basic Interface 2D tests
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen8and16)
{
    TestRoutine(8, 16, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen7and14)
{
    TestRoutine(7, 14, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen18and25)
{
    TestRoutine(18, 25, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen25and125)
{
    TestRoutine(25, 125, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen14and81)
{
    TestRoutine(14, 81, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen60and120)
{
    TestRoutine(60, 120, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen64and125)
{
    TestRoutine(64, 125, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen100and120)
{
    TestRoutine(100, 120, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen210and60)
{
    TestRoutine(210, 60, -1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, FwdLen864and25_quick)
{
    TestRoutine(864, 25, -1);
}

TEST_F(BasicInterfaceDouble2DBasisTest, InvLen8and16)
{
    TestRoutine(8, 16, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen7and14)
{
    TestRoutine(7, 14, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen18and25)
{
    TestRoutine(18, 25, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen25and125)
{
    TestRoutine(25, 125, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen14and81)
{
    TestRoutine(14, 81, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen60and120)
{
    TestRoutine(60, 120, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen64and125)
{
    TestRoutine(64, 125, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen100and120)
{
    TestRoutine(100, 120, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen210and60)
{
    TestRoutine(210, 60, 1);
}
TEST_F(BasicInterfaceDouble2DBasisTest, InvLen864and25_quick)
{
    TestRoutine(864, 25, 1);
}

TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen8and16)
{
    TestRoutine(8, 16, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen7and14)
{
    TestRoutine(7, 14, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen18and25)
{
    TestRoutine(18, 25, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen25and125)
{
    TestRoutine(25, 125, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen14and81)
{
    TestRoutine(14, 81, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen60and120)
{
    TestRoutine(60, 120, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen64and125)
{
    TestRoutine(64, 125, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen100and120)
{
    TestRoutine(100, 120, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen210and60)
{
    TestRoutine(210, 60, -1);
}

TEST_F(BasicInterfaceSingle2DBasisTest, InvLen8and16)
{
    TestRoutine(8, 16, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen7and14)
{
    TestRoutine(7, 14, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen18and25)
{
    TestRoutine(18, 25, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen25and125)
{
    TestRoutine(25, 125, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen14and81)
{
    TestRoutine(14, 81, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen60and120)
{
    TestRoutine(60, 120, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen64and125)
{
    TestRoutine(64, 125, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen100and120)
{
    TestRoutine(100, 120, 1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, InvLen210and60)
{
    TestRoutine(210, 60, 1);
}

TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen900and900)
{
    TestRoutine(900, 900, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen1296and1296)
{
    TestRoutine(1296, 1296, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen1000and1000)
{
    TestRoutine(1000, 1000, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen625and3125)
{
    TestRoutine(625, 3125, -1);
}
TEST_F(BasicInterfaceSingle2DBasisTest, FwdLen2187and729)
{
    TestRoutine(2187, 729, -1);
}

// Basic Interface 3D tests
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen8and16and8)
{
    TestRoutine(8, 16, 8, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen7and14and12)
{
    TestRoutine(7, 14, 12, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen18and25and32)
{
    TestRoutine(18, 25, 32, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen25and125and5)
{
    TestRoutine(25, 125, 5, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen14and81and15)
{
    TestRoutine(14, 81, 15, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen60and120and4)
{
    TestRoutine(60, 120, 4, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen64and125and27)
{
    TestRoutine(64, 125, 27, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen100and120and9)
{
    TestRoutine(100, 120, 9, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen210and60and36)
{
    TestRoutine(210, 60, 36, -1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, FwdLen210and60and64_quick)
{
    TestRoutine(210, 60, 64, -1);
}

TEST_F(BasicInterfaceDouble3DBasisTest, InvLen8and16and8)
{
    TestRoutine(8, 16, 8, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen7and14and12)
{
    TestRoutine(7, 14, 12, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen18and25and32)
{
    TestRoutine(18, 25, 32, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen25and125and5)
{
    TestRoutine(25, 125, 5, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen14and81and15)
{
    TestRoutine(14, 81, 15, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen60and120and4)
{
    TestRoutine(60, 120, 4, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen64and125and27)
{
    TestRoutine(64, 125, 27, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen100and120and9)
{
    TestRoutine(100, 120, 9, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen210and60and36)
{
    TestRoutine(210, 60, 36, 1);
}
TEST_F(BasicInterfaceDouble3DBasisTest, InvLen210and60and64_quick)
{
    TestRoutine(210, 60, 64, 1);
}

TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen8and16and8)
{
    TestRoutine(8, 16, 8, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen7and14and12)
{
    TestRoutine(7, 14, 12, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen18and25and32)
{
    TestRoutine(18, 25, 32, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen25and125and5)
{
    TestRoutine(25, 125, 5, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen14and81and15)
{
    TestRoutine(14, 81, 15, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen60and120and4)
{
    TestRoutine(60, 120, 4, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen64and125and27)
{
    TestRoutine(64, 125, 27, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen100and120and9)
{
    TestRoutine(100, 120, 9, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen210and60and36)
{
    TestRoutine(210, 60, 36, -1);
}

TEST_F(BasicInterfaceSingle3DBasisTest, InvLen8and16and8)
{
    TestRoutine(8, 16, 8, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen7and14and12)
{
    TestRoutine(7, 14, 12, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen18and25and32)
{
    TestRoutine(18, 25, 32, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen25and125and5)
{
    TestRoutine(25, 125, 5, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen14and81and15)
{
    TestRoutine(14, 81, 15, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen60and120and4)
{
    TestRoutine(60, 120, 4, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen64and125and27)
{
    TestRoutine(64, 125, 27, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen100and120and9)
{
    TestRoutine(100, 120, 9, 1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, InvLen210and60and36)
{
    TestRoutine(210, 60, 36, 1);
}

TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen100and100and100)
{
    TestRoutine(100, 100, 100, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen125and125and125)
{
    TestRoutine(125, 125, 125, -1);
}
TEST_F(BasicInterfaceSingle3DBasisTest, FwdLen243and243and243)
{
    TestRoutine(243, 243, 243, -1);
}

// Parameterized tests

TEST_P(BasicInterfaceDouble1DBasisTest, Fwd)
{
    TrLen1D t = GetParam();
    TestRoutine(t.N0, -1);
}
TEST_P(BasicInterfaceDouble1DBasisTest, Inv)
{
    TrLen1D t = GetParam();
    TestRoutine(t.N0, 1);
}
TEST_P(BasicInterfaceSingle1DBasisTest, Fwd)
{
    TrLen1D t = GetParam();
    TestRoutine(t.N0, -1);
}
TEST_P(BasicInterfaceSingle1DBasisTest, Inv)
{
    TrLen1D t = GetParam();
    TestRoutine(t.N0, 1);
}

TEST_P(BasicInterfaceDouble2DBasisTest, Fwd)
{
    TrLen2D t = GetParam();
    TestRoutine(t.N0, t.N1, -1);
}
TEST_P(BasicInterfaceDouble2DBasisTest, Inv)
{
    TrLen2D t = GetParam();
    TestRoutine(t.N0, t.N1, 1);
}
TEST_P(BasicInterfaceSingle2DBasisTest, Fwd)
{
    TrLen2D t = GetParam();
    TestRoutine(t.N0, t.N1, -1);
}
TEST_P(BasicInterfaceSingle2DBasisTest, Inv)
{
    TrLen2D t = GetParam();
    TestRoutine(t.N0, t.N1, 1);
}

TEST_P(BasicInterfaceDouble3DBasisTest, Fwd)
{
    TrLen3D t = GetParam();
    TestRoutine(t.N0, t.N1, t.N2, -1);
}
TEST_P(BasicInterfaceDouble3DBasisTest, Inv)
{
    TrLen3D t = GetParam();
    TestRoutine(t.N0, t.N1, t.N2, 1);
}
TEST_P(BasicInterfaceSingle3DBasisTest, Fwd)
{
    TrLen3D t = GetParam();
    TestRoutine(t.N0, t.N1, t.N2, -1);
}
TEST_P(BasicInterfaceSingle3DBasisTest, Inv)
{
    TrLen3D t = GetParam();
    TestRoutine(t.N0, t.N1, t.N2, 1);
}

INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceDouble1DBasisTest,
                        ::testing::ValuesIn(Test1DLengths));
INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceSingle1DBasisTest,
                        ::testing::ValuesIn(Test1DLengths));

INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceDouble2DBasisTest,
                        ::testing::ValuesIn(Test2DLengths));
INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceSingle2DBasisTest,
                        ::testing::ValuesIn(Test2DLengths));

INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceDouble3DBasisTest,
                        ::testing::ValuesIn(Test3DLengths));
INSTANTIATE_TEST_CASE_P(ListParamTest,
                        BasicInterfaceSingle3DBasisTest,
                        ::testing::ValuesIn(Test3DLengths));

#if 0

template <typename T, typename CT, typename P, void (*exec_plan)(P), void (*destruct_plan)(P)>
class AdvancedInterfaceBasisTest : public ::testing::Test
{
protected:

	virtual void TearDown(){}
	virtual void SetUp()
	{
		in = NULL;
		out = NULL;
		ref = NULL;
		p = NULL;
	}

	virtual void RunBvt(size_t L, size_t batch)
	{
		if(!SupportedLength(L))
			return;

		exec_plan(p);
		ErrorCheck<T,CT>(L, ref, out, batch*L);
	}

	CT *in, *out, *ref;
	P p;
};

template <typename T, typename CT, typename P, P (*planner)(int, const int *, int, CT *, const int *, int , int, CT *, const int *, int, int, int, unsigned int), void (*exec_plan)(P), void (*destruct_plan)(P)>
class AdvancedInterface1DBasisTest : public AdvancedInterfaceBasisTest <T, CT, P, exec_plan, destruct_plan>, public ::testing::WithParamInterface<TrLen1D>
{
protected:

	virtual void TestRoutine(size_t N, size_t batch, int dir)
	{
		int n[] = {(int)N};
		int *inembed = n;
		int *onembed = n;

		FftBasisVectorMixComplex<T, CT> bvm(1, &N, batch, FBC_RANDOM);
		bvm.RawPtrs(&this->in, &this->out);

		this->p = planner(1, n, (int)batch, this->in, inembed, 1, (int)N, this->out, onembed, 1, (int)N, dir, 0);

		bvm.Generate(&this->in, &this->ref, dir);

		this->RunBvt(N, batch);

		destruct_plan(this->p);
	}
};

template <typename T, typename CT, typename P, P (*planner)(int, const int *, int, CT *, const int *, int , int, CT *, const int *, int, int, int, unsigned int), void (*exec_plan)(P), void (*destruct_plan)(P)>
class AdvancedInterface2DBasisTest : public AdvancedInterfaceBasisTest <T, CT, P, exec_plan, destruct_plan>, public ::testing::WithParamInterface<TrLen2D>
{
protected:
	virtual void TestRoutine(size_t N0, size_t N1, size_t batch, int dir)
	{
		int n[] = {(int)N1, (int)N0};
		int *inembed = n;
		int *onembed = n;

		size_t L = N0*N1;
		size_t length[2];
		length[0] = N0;
		length[1] = N1;
		FftBasisVectorMixComplex<T, CT> bvm(2, length, batch, FBC_RANDOM);
		bvm.RawPtrs(&this->in, &this->out);

		this->p = planner(2, n, (int)batch, this->in, inembed, 1, (int)L, this->out, onembed, 1, (int)L, dir, 0);

		bvm.Generate(&this->in, &this->ref, dir);

		this->RunBvt(L, batch);

		destruct_plan(this->p);
	}
};

template <typename T, typename CT, typename P, P (*planner)(int, const int *, int, CT *, const int *, int , int, CT *, const int *, int, int, int, unsigned int), void (*exec_plan)(P), void (*destruct_plan)(P)>
class AdvancedInterface3DBasisTest : public AdvancedInterfaceBasisTest <T, CT, P, exec_plan, destruct_plan>, public ::testing::WithParamInterface<TrLen3D>
{
protected:
	virtual void TestRoutine(size_t N0, size_t N1, size_t N2, size_t batch, int dir)
	{
		int n[] = {(int)N2, (int)N1, (int)N0};
		int *inembed = n;
		int *onembed = n;

		size_t L = N0*N1*N2;
		size_t length[3];
		length[0] = N0;
		length[1] = N1;
		length[2] = N2;
		FftBasisVectorMixComplex<T, CT> bvm(3, length, batch, FBC_RANDOM);
		bvm.RawPtrs(&this->in, &this->out);

		this->p = planner(3, n, (int)batch, this->in, inembed, 1, (int)L, this->out, onembed, 1, (int)L, dir, 0);

		bvm.Generate(&this->in, &this->ref, dir);

		this->RunBvt(L, batch);

		destruct_plan(this->p);
	}
};

typedef AdvancedInterface1DBasisTest<double, complex_double, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceDouble1DBasisTest;
typedef AdvancedInterface2DBasisTest<double, complex_double, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceDouble2DBasisTest;
typedef AdvancedInterface3DBasisTest<double, complex_double, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceDouble3DBasisTest;

typedef AdvancedInterface1DBasisTest<float, complex_single, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceSingle1DBasisTest;
typedef AdvancedInterface2DBasisTest<float, complex_single, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceSingle2DBasisTest;
typedef AdvancedInterface3DBasisTest<float, complex_single, rocfft_plan, rocfft_plan_many_dft, rocfft_execute, rocfft_plan_destroy> AdvancedInterfaceSingle3DBasisTest;


TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen8 )	{ TestRoutine(8,   3,   -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen7 )	{ TestRoutine(7,   5,   -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen25 )	{ TestRoutine(25,  1,   -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen81 )	{ TestRoutine(81,  19,  -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen60 )	{ TestRoutine(60,  25,  -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen64 )	{ TestRoutine(64,  255, -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen162 )	{ TestRoutine(162, 16,  -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen100 )	{ TestRoutine(100, 64,  -1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, FwdLen210 )	{ TestRoutine(210, 33,  -1); }

TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen8 )	{ TestRoutine(8,   3,    1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen7 )	{ TestRoutine(7,   5,    1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen25 )	{ TestRoutine(25,  1,    1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen81 )	{ TestRoutine(81,  19,   1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen60 )	{ TestRoutine(60,  25,   1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen64 )	{ TestRoutine(64,  255,  1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen162 )	{ TestRoutine(162, 16,   1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen100 )	{ TestRoutine(100, 64,   1); }
TEST_F( AdvancedInterfaceDouble1DBasisTest, InvLen210 )	{ TestRoutine(210, 33,   1); }


TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen8 )	{ TestRoutine(8,   3,   -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen7 )	{ TestRoutine(7,   5,   -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen25 )	{ TestRoutine(25,  1,   -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen81 )	{ TestRoutine(81,  19,  -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen60 )	{ TestRoutine(60,  25,  -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen64 )	{ TestRoutine(64,  255, -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen162 )	{ TestRoutine(162, 16,  -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen100 )	{ TestRoutine(100, 64,  -1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, FwdLen210 )	{ TestRoutine(210, 33,  -1); }

TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen8 )	{ TestRoutine(8,   3,    1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen7 )	{ TestRoutine(7,   5,    1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen25 )	{ TestRoutine(25,  1,    1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen81 )	{ TestRoutine(81,  19,   1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen60 )	{ TestRoutine(60,  25,   1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen64 )	{ TestRoutine(64,  255,  1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen162 )	{ TestRoutine(162, 16,   1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen100 )	{ TestRoutine(100, 64,   1); }
TEST_F( AdvancedInterfaceSingle1DBasisTest, InvLen210 )	{ TestRoutine(210, 33,   1); }


// Advanced Interface 2D tests
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen8and16 )		{ TestRoutine(8,16,    200,  -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen7and14 )		{ TestRoutine(7,14,    1000, -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen18and25 )		{ TestRoutine(18,25,   750,  -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen25and125 )	{ TestRoutine(25,125,  11,   -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen14and81 )		{ TestRoutine(14,81,   91,   -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen60and120 )	{ TestRoutine(60,120,  17,   -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen64and125 )	{ TestRoutine(64,125,  13,   -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen100and120 )	{ TestRoutine(100,120, 12,   -1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, FwdLen210and60 )	{ TestRoutine(210,60,  8,    -1); }

TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen8and16 )		{ TestRoutine(8,16,    200,   1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen7and14 )		{ TestRoutine(7,14,    1000,  1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen18and25 )		{ TestRoutine(18,25,   750,   1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen25and125 )	{ TestRoutine(25,125,  11,    1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen14and81 )		{ TestRoutine(14,81,   91,    1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen60and120 )	{ TestRoutine(60,120,  17,    1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen64and125 )	{ TestRoutine(64,125,  13,    1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen100and120 )	{ TestRoutine(100,120, 12,    1); }
TEST_F( AdvancedInterfaceDouble2DBasisTest, InvLen210and60 )	{ TestRoutine(210,60,  8,     1); }


TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen8and16 )		{ TestRoutine(8,16,    200,  -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen7and14 )		{ TestRoutine(7,14,    1000, -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen18and25 )		{ TestRoutine(18,25,   750,  -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen25and125 )	{ TestRoutine(25,125,  11,   -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen14and81 )		{ TestRoutine(14,81,   91,   -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen60and120 )	{ TestRoutine(60,120,  17,   -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen64and125 )	{ TestRoutine(64,125,  13,   -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen100and120 )	{ TestRoutine(100,120, 12,   -1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, FwdLen210and60 )	{ TestRoutine(210,60,  8,    -1); }

TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen8and16 )		{ TestRoutine(8,16,    200,   1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen7and14 )		{ TestRoutine(7,14,    1000,  1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen18and25 )		{ TestRoutine(18,25,   750,   1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen25and125 )	{ TestRoutine(25,125,  11,    1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen14and81 )		{ TestRoutine(14,81,   91,    1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen60and120 )	{ TestRoutine(60,120,  17,    1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen64and125 )	{ TestRoutine(64,125,  13,    1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen100and120 )	{ TestRoutine(100,120, 12,    1); }
TEST_F( AdvancedInterfaceSingle2DBasisTest, InvLen210and60 )	{ TestRoutine(210,60,  8,     1); }




// Advance Interface 3D tests
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen8and16and8 )		{ TestRoutine(8,16,8,    128, -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen7and14and12 )		{ TestRoutine(7,14,12,   100, -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen18and25and32 )	{ TestRoutine(18,25,32,  85,  -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen25and125and5 )	{ TestRoutine(25,125,5,  70,  -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen14and81and15 )	{ TestRoutine(14,81,15,  41,  -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen60and120and4 )	{ TestRoutine(60,120,4,  22,  -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen64and125and27 )	{ TestRoutine(64,125,27, 13,  -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen100and120and9 )	{ TestRoutine(100,120,9, 8,   -1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, FwdLen210and60and36 )	{ TestRoutine(210,60,36, 3,   -1); }

TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen8and16and8 )		{ TestRoutine(8,16,8,    128,  1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen7and14and12 )		{ TestRoutine(7,14,12,   100,  1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen18and25and32 )	{ TestRoutine(18,25,32,  85,   1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen25and125and5 )	{ TestRoutine(25,125,5,  70,   1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen14and81and15 )	{ TestRoutine(14,81,15,  41,   1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen60and120and4 )	{ TestRoutine(60,120,4,  22,   1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen64and125and27 )	{ TestRoutine(64,125,27, 13,   1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen100and120and9 )	{ TestRoutine(100,120,9, 8,    1); }
TEST_F( AdvancedInterfaceDouble3DBasisTest, InvLen210and60and36 )	{ TestRoutine(210,60,36, 3,    1); }


TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen8and16and8 )		{ TestRoutine(8,16,8,    128, -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen7and14and12 )		{ TestRoutine(7,14,12,   100, -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen18and25and32 )	{ TestRoutine(18,25,32,  85,  -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen25and125and5 )	{ TestRoutine(25,125,5,  70,  -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen14and81and15 )	{ TestRoutine(14,81,15,  41,  -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen60and120and4 )	{ TestRoutine(60,120,4,  22,  -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen64and125and27 )	{ TestRoutine(64,125,27, 13,  -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen100and120and9 )	{ TestRoutine(100,120,9, 8,   -1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, FwdLen210and60and36 )	{ TestRoutine(210,60,36, 3,   -1); }

TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen8and16and8 )		{ TestRoutine(8,16,8,    128,  1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen7and14and12 )		{ TestRoutine(7,14,12,   100,  1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen18and25and32 )	{ TestRoutine(18,25,32,  85,   1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen25and125and5 )	{ TestRoutine(25,125,5,  70,   1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen14and81and15 )	{ TestRoutine(14,81,15,  41,   1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen60and120and4 )	{ TestRoutine(60,120,4,  22,   1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen64and125and27 )	{ TestRoutine(64,125,27, 13,   1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen100and120and9 )	{ TestRoutine(100,120,9, 8,    1); }
TEST_F( AdvancedInterfaceSingle3DBasisTest, InvLen210and60and36 )	{ TestRoutine(210,60,36, 3,    1); }


// Parameterized tests

TEST_P( AdvancedInterfaceDouble1DBasisTest, Fwd ) { TrLen1D t = GetParam(); TestRoutine(t.N0, 10, -1); }
TEST_P( AdvancedInterfaceDouble1DBasisTest, Inv ) { TrLen1D t = GetParam(); TestRoutine(t.N0, 29,   1); }
TEST_P( AdvancedInterfaceSingle1DBasisTest, Fwd ) { TrLen1D t = GetParam(); TestRoutine(t.N0, 19, -1); }
TEST_P( AdvancedInterfaceSingle1DBasisTest, Inv ) { TrLen1D t = GetParam(); TestRoutine(t.N0, 16,  1); }

TEST_P( AdvancedInterfaceDouble2DBasisTest, Fwd ) { TrLen2D t = GetParam(); TestRoutine(t.N0, t.N1, 7, -1); }
TEST_P( AdvancedInterfaceDouble2DBasisTest, Inv ) { TrLen2D t = GetParam(); TestRoutine(t.N0, t.N1, 6,  1); }
TEST_P( AdvancedInterfaceSingle2DBasisTest, Fwd ) { TrLen2D t = GetParam(); TestRoutine(t.N0, t.N1, 1, -1); }
TEST_P( AdvancedInterfaceSingle2DBasisTest, Inv ) { TrLen2D t = GetParam(); TestRoutine(t.N0, t.N1, 5,  1); }

TEST_P( AdvancedInterfaceDouble3DBasisTest, Fwd ) { TrLen3D t = GetParam(); TestRoutine(t.N0, t.N1, t.N2, 1, -1); }
TEST_P( AdvancedInterfaceDouble3DBasisTest, Inv ) { TrLen3D t = GetParam(); TestRoutine(t.N0, t.N1, t.N2, 2,  1); }
TEST_P( AdvancedInterfaceSingle3DBasisTest, Fwd ) { TrLen3D t = GetParam(); TestRoutine(t.N0, t.N1, t.N2, 3, -1); }
TEST_P( AdvancedInterfaceSingle3DBasisTest, Inv ) { TrLen3D t = GetParam(); TestRoutine(t.N0, t.N1, t.N2, 4,  1); }


INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceDouble1DBasisTest, ::testing::ValuesIn(Test1DLengths));
INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceSingle1DBasisTest, ::testing::ValuesIn(Test1DLengths));

INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceDouble2DBasisTest, ::testing::ValuesIn(Test2DLengths));
INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceSingle2DBasisTest, ::testing::ValuesIn(Test2DLengths));

INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceDouble3DBasisTest, ::testing::ValuesIn(Test3DLengths));
INSTANTIATE_TEST_CASE_P( ListParamTest, AdvancedInterfaceSingle3DBasisTest, ::testing::ValuesIn(Test3DLengths));

#endif

// int _tmain( int argc, _TCHAR* argv[ ] )
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    rocfft_setup();

    int retVal = RUN_ALL_TESTS();

    rocfft_cleanup();

    /*
      //  Reflection code to inspect how many tests failed in gTest
      ::testing::UnitTest& unitTest = *::testing::UnitTest::GetInstance( );

      unsigned int failedTests = 0;
      for( int i = 0; i < unitTest.total_test_case_count( ); ++i )
      {
          const ::testing::TestCase& testCase = *unitTest.GetTestCase( i );
          for( int j = 0; j < testCase.total_test_count( ); ++j )
          {
              const ::testing::TestInfo& testInfo = *testCase.GetTestInfo( j );
              if( testInfo.result( )->Failed( ) )
                  ++failedTests;
          }
      }

      //  Print helpful message at termination if we detect errors, to help
     users figure out what to do next
      if( failedTests )
      {
          std::cout << "\nFailed tests detected in test pass; please run test
     again with:" << std::endl;
          std::cout << "\t--gtest_filter=<XXX> to select a specific failing test
     of interest" << std::endl;
          std::cout << "\t--gtest_catch_exceptions=0 to generate minidump of
     failing test, or" << std::endl;
          std::cout << "\t--gtest_break_on_failure to debug interactively with
     debugger" << std::endl;
          std::cout << "\t    (only on googletest assertion failures, not SEH
     exceptions)" << std::endl;
      }
  */

    return retVal;
}
