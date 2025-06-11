.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation, reference, environment variable, environment

********************************************************************
rocFFT environment variables
********************************************************************

This section describes the most important rocFFT environment variables,
which are grouped by functionality.

Logging and debugging
=====================

The logging and debugging environment variables for rocFFT are collected in the
following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCFFT_LAYER``
        | Controls which logging layers are active using bitmask values.
      - | ``0``: No logging
        | ``1``: Trace logging
        | ``2``: Bench logging
        | ``4``: Profile logging
        | ``8``: Plan logging
        | ``16``: Kernel I/O logging
        | ``32``: RTC logging
        | ``64``: Tuning logging
        | ``128``: Graph logging
        | Values can be combined by adding multiple values together
        | E.g. ``ROCFFT_LAYER=12`` enables plan and profile logging (8 + 4 = 12)

    * - | ``ROCFFT_LOG_TRACE_PATH``
        | Specifies file path for trace logging output.
      - | String path to trace log file
        | Used when trace logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_BENCH_PATH``
        | Specifies file path for bench logging output.
      - | String path to bench log file
        | Used when bench logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_PROFILE_PATH``
        | Specifies file path for profile logging output.
      - | String path to profile log file
        | Used when profile logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_PLAN_PATH``
        | Specifies file path for plan logging output.
      - | String path to plan log file
        | Used when plan logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_KERNELIO_PATH``
        | Specifies file path for kernel I/O logging output.
      - | String path to kernel I/O log file
        | Used when kernel I/O logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_RTC_PATH``
        | Specifies file path for runtime compilation logging output.
      - | String path to RTC log file
        | Used when RTC logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_TUNING_PATH``
        | Specifies file path for tuning logging output.
      - | String path to tuning log file
        | Used when tuning logging is enabled via ``ROCFFT_LAYER``

    * - | ``ROCFFT_LOG_GRAPH_PATH``
        | Specifies file path for graph logging output.
      - | String path to graph log file
        | Used when graph logging is enabled via ``ROCFFT_LAYER``
        | Output can be used with ``graphviz``

    * - | ``ROCFFT_DEVICE_BW``
        | Specifies device memory bandwidth for performance calculations.
      - | Numeric value representing device bandwidth in GB/s
        | Used when computing numbers for profile logging (``ROCFFT_LAYER=4``)

Runtime compilation and caching
===============================

The runtime compilation and caching environment variables for rocFFT are collected in
the following table. For more information, see :doc:`Use runtime compilation <../how-to/runtime-compilation>`.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCFFT_RTC_CACHE_PATH``
        | Sets the location for the read-write user-level cache of compiled kernels.
      - | String path to writable file location
        | Enables persistent caching across program runs

    * - | ``ROCFFT_RTC_CACHE_READ_DISABLE``
        | Disables reading from the runtime compilation cache.
      - | ``0`` or unset: Enable cache reading
        | ``1``: Disable cache reading

    * - | ``ROCFFT_RTC_CACHE_WRITE_DISABLE``
        | Disables writing to the runtime compilation cache.
      - | ``0`` or unset: Enable cache writing
        | ``1``: Disable cache writing

    * - | ``ROCFFT_RTC_PROCESS_HELPER``
        | Specifies path to custom RTC helper process executable.
      - | String path to RTC helper executable
        | Used for out-of-process kernel compilation

Solution map and configuration
==============================

The solution map and configuration environment variables for rocFFT are collected in the
following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCFFT_READ_EXPLICIT_SOL_MAP_FILE``
        | Specifies explicit solution map file to read.
      - | String path to solution map file
        | Overrides default solution map discovery

    * - | ``ROCFFT_READ_SOL_MAP_FROM_FOLDER``
        | Specifies folder containing solution map files.
      - | String path to folder with solution map files
        | Loads architecture-specific and generic solution maps

Development and debugging (advanced)
====================================

The development and debugging environment variables for rocFFT are collected in the
following table. These variables are primarily intended for debugging and development
purposes.

.. note::

   These variables are very complicated to configure and are mainly designed for use
   by the rocFFT development team.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS``
        | Enables generation of standalone kernel test harnesses.
      - | ``0`` or unset: Don't generate harness files
        | ``1``: Generate kernel harness files

    * - | ``ROCFFT_DEBUG_KERNEL_HARNESS_PATH``
        | Specifies output directory for generated kernel harness files.
      - | String path for kernel harness output directory
        | Default: current working directory
