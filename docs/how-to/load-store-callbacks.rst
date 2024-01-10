.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _load-store-callbacks:

********************************************************************
Load and Store Callbacks
********************************************************************

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
  when the callback is set.  Shared memory is not supported,
  and this parameter is always null.

Callback functions are called exactly once for each element being
loaded or stored in a transform.  Note that multiple kernels may be
launched to decompose a transform, which means that separate kernels
may call the load and store callbacks for a transform if both are
specified.

Callbacks functions are only supported for transforms that do not use planar format for input or output.

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
