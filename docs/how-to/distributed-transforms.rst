.. _distributed-transforms:

********************************************************************
Distributed transforms
********************************************************************

rocFFT can optionally distribute FFTs across multiple devices in a
single process, or across multiple MPI ranks.  To perform distributed
transforms, users must describe their input and output data layouts
as :ref:`fields<input_output_fields>`.

Multiple devices in a single process
====================================

A transform may be distributed across multiple devices in a single
process by passing distinct device IDs to
:cpp:func:`rocfft_brick_create` for bricks in the input and output
fields.

Support for single-process multi-device transforms was introduced in
ROCm 6.0 with rocFFT 1.0.25.

Message Passing Interface (MPI)
===============================

MPI allows for distributing the transform across multiple processes,
organized into MPI ranks.

Support for MPI transforms was introduced in ROCm 6.3 with rocFFT
1.0.29.

.. note::

   rocFFT MPI support is only available when the library is built
   with the `ROCFFT_MPI_ENABLE` CMake option enabled.  By default it
   is off.

   Additionally, rocFFT MPI support requires a GPU-aware MPI library
   with support for transferring data to/from HIP devices.

   Finally, rocFFT API calls made on different ranks may return
   different values.  Users must take care to ensure that all ranks
   have successfully created their plans before attempting to execute
   a distributed transform, and it is possible for one rank to fail
   to create/execute a plan while the others succeed.

To perform a transform across multiple MPI ranks, additional steps
are required to distribute the computation:

#. Each rank calls :cpp:func:`rocfft_plan_description_set_comm` to
   add an MPI communicator to an allocated plan description.  rocFFT
   will distribute the computation across all ranks in the
   communicator.

#. Each rank allocates the same fields and calls
   :cpp:func:`rocfft_plan_description_add_infield` and
   :cpp:func:`rocfft_plan_description_add_outfield` on the plan
   description.  However, each rank must only call
   :cpp:func:`rocfft_brick_create` and
   :cpp:func:`rocfft_field_add_brick` for bricks that reside on that
   rank.

   A brick resides on exactly one rank, but each rank may have zero
   or more bricks on it.

#. Each rank in the communicator calls
   :cpp:func:`rocfft_plan_create`.  At this time rocFFT will distribute
   the supplied brick information between all of the ranks.

#. Each rank in the communicator calls :cpp:func:`rocfft_execute`.
   This function accepts arrays of pointers for input and output.
   The arrays contain pointers to each brick in the input or output
   of the current rank.

   The pointers must be provided in the same order in which the bricks were
   added to the field (via calls to :cpp:func:`rocfft_field_add_brick`), and
   must point to memory on the device that was specified at that time.

   For in-place transforms, only pass input pointers and pass an
   empty array of output pointers.
