
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

======
rocFFT
======

Introduction
------------

The rocFFT library is an implementation of discrete Fast Fourier Transforms written in HiP for GPU devices. The library:

* Provides a fast and accurate platform for calculating discrete FFTs.
* Supports in-place or out-of-place transforms.
* Supports 1D, 2D, and 3D transforms.
* Supports computation of transforms in batches.
* Supports real and complex FFTs.
* Supports lengths that are any combination of powers of 2, 3, 5.
* Supports single and double precision floating point formats.

FFT
---

The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the FFT definition to
reduce the mathematical intensity required from :math:`O(N^2)` to :math:`O(N \log N)` when the sequence length, *N*, is
the product of small prime factors.

Computation
-----------

What is computed by the library? Here are the formulas:

For a 1D complex DFT:

:math:`{\tilde{x}}_j = {{1}\over{scale}}\sum_{k=0}^{n-1}x_k\exp\left({\pm i}{{2\pi jk}\over{n}}\right)\hbox{ for } j=0,1,\ldots,n-1`

where, :math:`x_k` are the complex data to be transformed, :math:`\tilde{x}_j` are the transformed data, and the sign :math:`\pm`
determines the direction of the transform: :math:`-` for forward and :math:`+` for backward. Note that you must provide the scaling
factor.  By default, the scale is set to 1 for the transforms.

For a 2D complex DFT:

:math:`{\tilde{x}}_{jk} = {{1}\over{scale}}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rq}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1`, where, :math:`x_{rq}` are the complex data to be transformed,
:math:`\tilde{x}_{jk}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.  By default, the
scale is set to 1 for the transforms.

For a 3D complex DFT:

:math:`\tilde{x}_{jkl} = {{1}\over{scale}}\sum_{s=0}^{p-1}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rqs}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)\exp\left({\pm i}{{2\pi ls}\over{p}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1\hbox{ and } l=0,1,\ldots,p-1`, where :math:`x_{rqs}` are the complex data to
be transformed, :math:`\tilde{x}_{jkl}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.
By default, the scale is set to 1 for the transforms.

Setup and Cleanup of rocFFT
---------------------------

At the beginning of the program, before any of the library functions are called, the api :cpp:func:`rocfft_setup` has to be called. Similarly,
the function :cpp:func:`rocfft_cleanup` has to be called at the end of the program. These apis ensure resources are properly allocated and freed. 

Workflow
--------

In order to compute an FFT transform with rocFFT, a plan has to be created first. A plan is a handle to an internal data structure that
holds all the details about the transform that the user wishes to compute. After the plan is created, it can be executed (a separate api call) 
with the specified data buffers. The execution step can be repeated any number of times with the same plan on different input/output buffers
as needed. And when the plan is no longer needed, it gets destroyed.

Example
-------

.. code-block:: c

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "hip/hip_vector_types.h"
   #include "rocfft.h"
   
   int main()
   {
           // rocFFT gpu compute
           // ========================================
  
           rocfft_setup();

           size_t N = 16;
           size_t Nbytes = N * sizeof(float2);
   
           // Create HIP device buffer
           float2 *x;
           hipMalloc(&x, Nbytes);
   
           // Initialize data
           std::vector<float2> cx(N);
           for (size_t i = 0; i < N; i++)
           {
                   cx[i].x = 1;
                   cx[i].y = -1;
           }
   
           //  Copy data to device
           hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
   
           // Create rocFFT plan
           rocfft_plan plan = NULL;
           size_t length = N;
           rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, &length, 1, NULL);
   
           // Execute plan
           rocfft_execute(plan, (void**) &x, NULL, NULL);
   
           // Wait for execution to finish
           hipDeviceSynchronize();
   
           // Destroy plan
           rocfft_plan_destroy(plan);
   
           // Copy result back to host
           std::vector<float2> y(N);
           hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);
   
           // Print results
           for (size_t i = 0; i < N; i++)
           {
                   std::cout << y[i].x << ", " << y[i].y << std::endl;
           }
   
           // Free device buffer
           hipFree(x);
   
           rocfft_cleanup();

           return 0;
   }

Topic
-----

This is next topic

