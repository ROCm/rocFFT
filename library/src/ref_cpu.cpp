#include "kernel_launch.h"
#include "ref_cpu.h"

/*
void compute_general_1d_fft(const void *data_p, void *back_p)
{
    RefLibHandle &refHandle = RefLibHandle::GetRefLibHandle();

    DeviceCallIn *data = (DeviceCallIn *)data_p;

    int n[1];
    n[0] = data->node->length[0];     
    int N = n[0];

	ftype_fftwf_malloc local_fftwf_malloc = (ftype_fftwf_malloc)dlsym(refHandle.fftw3f_lib, "fftwf_malloc");
	ftype_fftwf_free local_fftwf_free = (ftype_fftwf_free)dlsym(refHandle.fftw3f_lib, "fftwf_free");

    ftype_fftwf_plan_many_dft local_fftwf_plan_many_dft = (ftype_fftwf_plan_many_dft)dlsym(refHandle.fftw3f_lib, "fftwf_plan_many_dft");
    ftype_fftwf_execute local_fftwf_execute = (ftype_fftwf_execute)dlsym(refHandle.fftw3f_lib, "fftwf_execute");
    ftype_fftwf_destroy_plan local_fftwf_destroy_plan = (ftype_fftwf_destroy_plan)dlsym(refHandle.fftw3f_lib, "fftwf_destroy_plan");

    //printf("\n%p", local_fftwf_malloc);
    //printf("\n%p", local_fftwf_free);

    local_fftwf_complex *in = (local_fftwf_complex *)local_fftwf_malloc(N * sizeof(local_fftwf_complex));
    local_fftwf_complex *ot = (local_fftwf_complex *)local_fftwf_malloc(N * sizeof(local_fftwf_complex));

    std::cout << std::endl;
    for(int i=0; i<N; i++)
    {
        in[i][0] = (float)i;
        in[i][1] = 0;

        ot[i][0] = 0;
        ot[i][1] = 0;
    }

    void *p = local_fftwf_plan_many_dft(1, n, 1, in, NULL, 1, 0, ot, NULL, 1, 0, LOCAL_FFTW_FORWARD, LOCAL_FFTW_ESTIMATE);
    local_fftwf_execute(p);
    local_fftwf_destroy_plan(p);

    for(int i=0; i<N; i++)
    {
        printf("(%f, %f)\n", ot[i][0], ot[i][1]);
    }

    local_fftwf_free(in);
    local_fftwf_free(ot);

    std::cout << std::endl;
}


void rocfft_internal_cpu_reference_op(const void *data_p, void *back_p)
{
    DeviceCallIn *data = (DeviceCallIn *)data_p;

    if(data->node->scheme == CS_KERNEL_STOCKHAM)
    {
        compute_general_1d_fft(data_p, back_p);
    }

    if(data->node->scheme == CS_KERNEL_TRANSPOSE)
    {
    }

    return;
}
*/


