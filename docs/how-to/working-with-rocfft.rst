.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _working-with-rocfft:

********************************************************************
Working with rocFFT
********************************************************************

Workflow
========

In order to compute an FFT with rocFFT, a plan has to be created first. A plan is a handle to an internal data structure that
holds the details about the transform that the user wishes to compute. After the plan is created, it can be executed (a separate API call)
with the specified data buffers. The execution step can be repeated any number of times with the same plan on different input/output buffers
as needed. And when the plan is no longer needed, it gets destroyed.

To do a transform,

#. Initialize the library by calling :cpp:func:`rocfft_setup()`.
#. Create a plan, for each distinct type of FFT needed:

   * To create a plan, do either of the following

     * If the plan specification is simple, call :cpp:func:`rocfft_plan_create` and specify the value of the fundamental parameters.
     * If the plan has more details, first a plan description is created with :cpp:func:`rocfft_plan_description_create`, and additional APIs such
       as :cpp:func:`rocfft_plan_description_set_data_layout` are called to specify plan details. And then, :cpp:func:`rocfft_plan_create` is called
       with the description handle passed to it along with other details.

   * Optionally, allocate a work buffer for the plan:

     * Call :cpp:func:`rocfft_plan_get_work_buffer_size` to check the size of work buffer required by the plan.
     * If a non-zero size is required:

       * Create an execution info object with :cpp:func:`rocfft_execution_info_create`.
       * Allocate a buffer using :cpp:func:`hipMalloc` and pass the allocated buffer to :cpp:func:`rocfft_execution_info_set_work_buffer`.

#. Execute the plan:

   * The execution API :cpp:func:`rocfft_execute` is used to do the actual computation on the data buffers specified.
   * Extra execution information such as work buffers and compute streams are passed to :cpp:func:`rocfft_execute` in the :cpp:type:`rocfft_execution_info` object.
   * :cpp:func:`rocfft_execute` can be called repeatedly as needed for different data, with the same plan.
   * If the plan requires a work buffer but none was provided, :cpp:func:`rocfft_execute` will automatically allocate a work buffer and free it when execution is finished.

#. If a work buffer was allocated:

   * Call :cpp:func:`hipFree` to free the work buffer.
   * Call :cpp:func:`rocfft_execution_info_destroy` to destroy the execution info object.

#. Destroy the plan by calling :cpp:func:`rocfft_plan_destroy`.
#. Terminate the library by calling :cpp:func:`rocfft_cleanup()`.


Example
=======

.. code-block:: c

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "hip/hip_vector_types.h"
   #include "rocfft/rocfft.h"
   
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
           rocfft_plan plan = nullptr;
           size_t length = N;
           rocfft_plan_create(&plan, rocfft_placement_inplace,
                rocfft_transform_type_complex_forward, rocfft_precision_single,
                1, &length, 1, nullptr);

	   // Check if the plan requires a work buffer
	   size_t work_buf_size = 0;
	   rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
	   void* work_buf = nullptr;
	   rocfft_execution_info info = nullptr;
	   if(work_buf_size)
           {
                   rocfft_execution_info_create(&info);
		   hipMalloc(&work_buf, work_buf_size);
		   rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
           }
   
           // Execute plan
           rocfft_execute(plan, (void**) &x, nullptr, info);
   
           // Wait for execution to finish
           hipDeviceSynchronize();

	   // Clean up work buffer
	   if(work_buf_size)
	   {
	           hipFree(work_buf);
		   rocfft_execution_info_destroy(info);
	   }

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

Library Setup and Cleanup
=========================

At the beginning of the program, before any of the library APIs are called, the function :cpp:func:`rocfft_setup` has to be called. Similarly,
the function :cpp:func:`rocfft_cleanup` has to be called at the end of the program. These APIs ensure resources are properly allocated and freed.

Plans
=====

A plan is the collection of (almost) all the parameters needed to specify an FFT computation. A rocFFT plan includes the
following information:

* Type of transform (complex or real)
* Dimension of the transform (1D, 2D, or 3D)
* Length or extent of data in each dimension
* Number of datasets that are transformed (batch size)
* Floating-point precision of the data
* In-place or not in-place transform
* Format (array type) of the input/output buffer
* Layout of data in the input/output buffer 
* Scaling factor to apply to the output of the transform

The rocFFT plan does not include the following parameters:

* The handles to the input and output data buffers.
* The handle to a temporary work buffer (if needed).
* Other information to control execution on the device.

These parameters are specified when the plan is executed.

Data
====

The input/output buffers that hold the data for the transform must be allocated, initialized and specified to the library by the
user. For larger transforms, temporary work buffers may be needed. Because the library tries to minimize its own allocation of
memory regions on the device, it expects the user to manage work buffers. The size of the buffer needed can be queried using
:cpp:func:`rocfft_plan_get_work_buffer_size` and after their allocation can be passed to the library by
:cpp:func:`rocfft_execution_info_set_work_buffer`. The samples in the source repository show how to use these.

Transform and Array types 
=========================

There are two main types of FFTs in the library:

* Complex FFT - Transformation of complex data (forward or backward); the library supports the following two
  array types to store complex numbers:

  #. Planar format - where the real and imaginary components are kept in 2 separate arrays:

     * Buffer1: ``RRRRR...`` 
     * Buffer2: ``IIIII...``
  #. Interleaved format - where the real and imaginary components are stored as contiguous pairs in the same array: 

     * Buffer: ``RIRIRIRIRIRI...``
  
* Real FFT - Transformation of real data. For transforms involving real data, there are two possibilities:

  * Real data being subject to forward FFT that results in complex data (Hermitian).
  * Complex data (Hermitian) being subject to backward FFT that results in real data.

.. note::

   Real backward FFTs require that the input data be
   Hermitian-symmetric, as would naturally happen in the output of a
   real forward FFT.  rocFFT will produce undefined results if
   this requirement is not met.

The library provides the :cpp:enum:`rocfft_transform_type` and
:cpp:enum:`rocfft_array_type` enums to specify transform and array
types, respectively.

Batches
=======

The efficiency of the library is improved by utilizing transforms in batches. Sending as much data as possible in a single
transform call leverages the parallel compute capabilities of devices (GPU devices in particular), and minimizes the penalty
of control transfer. It is best to think of a device as a high-throughput, high-latency device. Using a networking analogy as
an example, this approach is similar to having a massively high-bandwidth pipe with very high ping response times. If the client
is ready to send data to the device for compute, it should be sent in as few API calls as possible, and this can be done by batching.
rocFFT plans have a parameter `number_of_transforms` (this value is also referred to as batch size in various places in the document)
in :cpp:func:`rocfft_plan_create` to describe the number of transforms being requested. All 1D, 2D, and 3D transforms can be batched.

.. _resultplacement:

Result placement
================

The API supports both in-place and not in-place transforms via the :cpp:enum:`rocfft_result_placement` enum.  With in-place transforms, only input buffers are provided to the
execution API, and the resulting data is written to the same buffer, overwriting the input data.  With not in-place transforms, distinct
output buffers are provided, and the results are written into the output buffer.

.. note::

   rocFFT may overwrite input buffers on real inverse (complex-to-real) transforms, even if they are requested to not be in-place.  rocFFT is able to optimize the FFT better by doing this.

.. _stridesdistances:

Strides and Distances
=====================

Strides and distances enable users to specify custom layout of data using :cpp:func:`rocfft_plan_description_set_data_layout`.

For 1D data, if :cpp:expr:`strides[0] == strideX == 1`, successive elements in the first dimension (dimension index 0) are stored
contiguously in memory. If :cpp:expr:`strideX` is a value greater than 1, gaps in memory exist between each element of the vector.
For multidimensional cases; if :cpp:expr:`strides[1] == strideY == LenX` for 2D data and :cpp:expr:`strides[2] == strideZ == LenX * LenY` for 3D data,
no gaps exist in memory between each element, and all vectors are stored tightly packed in memory. Here, :cpp:expr:`LenX`, :cpp:expr:`LenY`, and :cpp:expr:`LenZ` denote the
transform lengths :cpp:expr:`lengths[0]`, :cpp:expr:`lengths[1]`, and :cpp:expr:`lengths[2]`, respectively, which are used to set up the plan.

Distance is the stride that exists between corresponding elements of successive FFT data instances (primitives) in a batch. Distance is measured in units of the memory type;
complex data measures in complex units, and real data measures in real units. For tightly packed data, the distance between FFT primitives is the size of the FFT primitive,
such that :cpp:expr:`dist == LenX` for 1D data, :cpp:expr:`dist == LenX * LenY` for 2D data, and :cpp:expr:`dist == LenX * LenY * LenZ` for 3D data. It is possible to set the distance of a plan to be less than the size
of the FFT vector; typically 1 when doing column (strided) access on packed data. When computing a batch of 1D FFT vectors, if :cpp:expr:`distance == 1`, and :cpp:expr:`strideX == length(vector)`,
it means data for each logical FFT is read along columns (in this case along the batch). You must verify that the distance and strides are valid, such that each logical
FFT instance is not overlapping with any other on output data; if not valid, undefined results may occur. Overlapping on input data is generally allowed. 
A simple example of a column data access pattern would be to perform a 1D length 4096 on each row of an array of 1024 rows x 4096 columns of values stored in a column-major array, 
such as a FORTRAN program might provide. (This would be equivalent to a C or C++ program that has an array of 4096 rows x 1024 columns stored in a row-major manner, on which you want 
to perform a 1D length 4096 transform on each column.) In this case, specify the strides as [1024] and distance as 1.

Overwriting non-contiguous buffers
==================================

rocFFT guarantees that both the reading of FFT input and the writing of FFT output will respect the
specified strides.  However, temporary results can potentially be written to these buffers
contiguously, which may be unexpected if the strides would avoid certain memory locations completely
for reading and writing.

For example, a 1D FFT of length :math:`N` with input and output stride of 2 is transforming only
even-indexed elements in the input and output buffers.  But if temporary data needs to be written to
the buffers, odd-indexed elements may be overwritten.

However, rocFFT is guaranteed to respect the size of buffers.  In the above example, the
input/output buffers are :math:`2N` elements long, even if only :math:`N` even-indexed
elements are being transformed.  No more than :math:`2N` elements of temporary data will be written
to the buffers during the transform.

These policies apply to both input and output buffers, because :ref:`not in-place transforms may overwrite input data<resultplacement>`.

Input and Output Fields
=======================

By default, rocFFT inputs and outputs are on the same device, and the layouts of
each are described using a set of strides passed to
:cpp:func:`rocfft_plan_description_set_data_layout`.

rocFFT optionally allows for inputs and outputs to be described as **fields**,
each of which is decomposed into multiple **bricks**, where each brick can
reside on a different device and have its own layout parameters.

.. note::

   The rocFFT APIs to declare fields and bricks are currently experimental and
   subject to change in future releases.  We welcome feedback and questions
   about these interfaces.  Please open issues on the `rocFFT issue tracker
   <https://github.com/ROCmSoftwarePlatform/rocFFT/issues>`_ with questions and
   comments.

The workflow for fields is as follows:

#. Allocate a :cpp:type:`rocfft_field` struct by calling :cpp:func:`rocfft_field_create`.

#. Add one or more bricks to the field:

   #. Allocate a :cpp:type:`rocfft_brick` by calling
      :cpp:func:`rocfft_brick_create`.  The brick's dimensions are
      defined in terms of lower and upper coordinates in the field's
      index space.

      Note that the lower coordinate is inclusive (contained within the brick) and
      the upper coordinate is exclusive (first index past the end of the brick).

      The device on which the brick resides is also specified at this time, along
      with the strides of the brick in device memory.

      All coordinates and strides provided here also include batch dimensions.

   #. Add the brick to the field by calling :cpp:func:`rocfft_field_add_brick`.

   #. Deallocate the brick by calling :cpp:func:`rocfft_brick_destroy`.

#. Set the field as an input or output for the transform by calling either
   :cpp:func:`rocfft_plan_description_add_infield` or
   :cpp:func:`rocfft_plan_description_add_outfield` on a plan description that
   has already been allocated.  The plan description must then be provided to
   :cpp:func:`rocfft_plan_create`.

   Offsets, strides, and distances as specified by
   :cpp:func:`rocfft_plan_description_set_data_layout` for input or output are
   ignored when a field is set for the corresponding input or output.

   If the same field layout is used for both input and output, the same
   :cpp:type:`rocfft_field` struct may be passed to both
   :cpp:func:`rocfft_plan_description_add_infield` and
   :cpp:func:`rocfft_plan_description_add_outfield`.

   For in-place transforms, only call
   :cpp:func:`rocfft_plan_description_add_infield` and do not call
   :cpp:func:`rocfft_plan_description_add_outfield`.

#. Deallocate the field by calling :cpp:func:`rocfft_field_destroy`.

#. Create the plan by calling :cpp:func:`rocfft_plan_create`.  Pass the plan
   description that has already been allocated.

#. Execute the plan by calling :cpp:func:`rocfft_execute`.  This function takes
   arrays of pointers for input and output.  If fields have been set for input
   or output, then the arrays must contain pointers to each brick in the input
   or output.

   The pointers must be provided in the same order in which the bricks were
   added to the field (via calls to :cpp:func:`rocfft_field_add_brick`), and
   must point to memory on the device that was specified at that time.

   For in-place transforms, only pass input pointers and do not pass output
   pointers.

Transforms of real data
=======================

.. toctree::
   :maxdepth: 2

   real

Reproducibility of Results
==========================

The results of an FFT computation generated by rocFFT are bitwise reproducible,
i.e., deterministic behavior is expected between runs, generating the exact same
result at every run. Bitwise reproducibility is achieved as long as the following
is kept constant between runs:

 * FFT parameters
 * rocFFT library version
 * GPU model

A valid FFT plan is a requirement for reproducibility; in particular, the
:ref:`rules for overlapping of FFT data<stridesdistances>` must be followed.


Result Scaling
==============

The output of a forward or backward FFT often needs to be multiplied
by a scaling factor before the data can be passed to the next step of
a computation.  While users of rocFFT can launch a separate GPU
kernel to do this work, rocFFT provides a
:cpp:func:`rocfft_plan_description_set_scale_factor` function to more
efficiently combine this scaling multiplication with the FFT work.

The scaling factor is set on the plan description prior to plan creation.

Load and Store Callbacks
========================

rocFFT includes experimental functionality to call user-defined device functions
when loading input from global memory at the start of a transform, or
when storing output to global memory at the end of a transform.

These user-defined callback functions may be optionally supplied
to the library using
:cpp:func:`rocfft_execution_info_set_load_callback` and
:cpp:func:`rocfft_execution_info_set_store_callback`.

Device functions supplied as callbacks must load and store element
data types that are appropriate for the transform being performed.

+-------------------------+--------------------+----------------------+
|Transform type           | Load element type  | Store element type   |
+=========================+====================+======================+
|Complex-to-complex,      | `_Float16_2`       | `_Float16_2`         |
|half-precision           |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-complex,      | `float2`           | `float2`             |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-complex,      | `double2`          | `double2`            |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Real-to-complex,         | `float`            | `float2`             |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Real-to-complex,         | `_Float16`         | `_Float16_2`         |
|half-precision           |                    |                      |
+-------------------------+--------------------+----------------------+
|Real-to-complex,         | `double`           | `double2`            |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-real,         | `_Float16_2`       | `_Float16`           |
|half-precision           |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-real,         | `float2`           | `float`              |
|single-precision         |                    |                      |
+-------------------------+--------------------+----------------------+
|Complex-to-real,         | `double2`          | `double`             |
|double-precision         |                    |                      |
+-------------------------+--------------------+----------------------+

The callback function signatures must match the specifications
below.

.. code-block:: c

  T load_callback(T* buffer, size_t offset, void* callback_data, void* shared_memory);
  void store_callback(T* buffer, size_t offset, T element, void* callback_data, void* shared_memory);

The parameters for the functions are defined as:

* `T`: The data type of each element being loaded or stored from the
  input or output.
* `buffer`: Pointer to the input (for load callbacks) or
  output (for store callbacks) in device memory that was passed to
  :cpp:func:`rocfft_execute`.
* `offset`: The offset of the location being read from or written
  to.  This counts in elements, from the `buffer` pointer.
* `element`: For store callbacks only, the element to be stored.
* `callback_data`: A pointer value accepted by
  :cpp:func:`rocfft_execution_info_set_load_callback` and
  :cpp:func:`rocfft_execution_info_set_store_callback` which is passed
  through to the callback function.
* `shared_memory`: A pointer to an amount of shared memory requested
  when the callback is set.  Currently, shared memory is not supported
  and this parameter is always null.

Callback functions are called exactly once for each element being
loaded or stored in a transform.  Note that multiple kernels may be
launched to decompose a transform, which means that separate kernels
may call the load and store callbacks for a transform if both are
specified.

Currently, callbacks functions are only supported for transforms that
do not use planar format for input or output.

Runtime compilation
===================

rocFFT includes many kernels for common FFT problems.  Some plans may
require additional kernels aside from what is built in to the
library.  In these cases, rocFFT will compile optimized kernels for
the plan when the plan is created.

Compiled kernels are stored in memory by default and will be reused
if they are required again for plans in the same process.

If the ``ROCFFT_RTC_CACHE_PATH`` environment variable is set to a
writable file location, rocFFT will write compiled kernels to this
location.  rocFFT will read kernels from this location for plans in
other processes that need runtime-compiled kernels.  rocFFT will
create the specified file if it does not already exist.
