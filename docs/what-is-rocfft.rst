.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _what-is-rocfft:

********************************************************************
What is rocFFT?
********************************************************************

Introduction
============

The rocFFT library is an implementation of the discrete Fast Fourier Transform (FFT) written in HIP for GPU devices.
The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocFFT

The rocFFT library provides a fast and accurate platform for calculating discrete FFTs. It supports the following features: 

* Half (FP16), single, and double precision floating point formats
* 1D, 2D, and 3D transforms
* Computation of transforms in batches
* Real and complex FFTs
* Arbitrary lengths, with optimizations for combinations of powers of 2, 3, 5, 7, 11, 13, and 17

FFT Computation
===============

The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the DFT definition to
reduce the mathematical complexity from :math:`O(N^2)` to :math:`O(N \log N)`.

What is computed by the library? Here are the formulas:

For a 1D complex DFT:

:math:`{\tilde{x}}_j = \sum_{k=0}^{n-1}x_k\exp\left({\pm i}{{2\pi jk}\over{n}}\right)\hbox{ for } j=0,1,\ldots,n-1`

Where, :math:`x_k` are the complex data to be transformed, :math:`\tilde{x}_j` are the transformed data, and the sign :math:`\pm`
determines the direction of the transform: :math:`-` for forward and :math:`+` for backward.

For a 2D complex DFT:

:math:`{\tilde{x}}_{jk} = \sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rq}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)`

For :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1`, where, :math:`x_{rq}` are the complex data to be transformed,
:math:`\tilde{x}_{jk}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.

For a 3D complex DFT:

:math:`\tilde{x}_{jkl} = \sum_{s=0}^{p-1}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rqs}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)\exp\left({\pm i}{{2\pi ls}\over{p}}\right)`

For :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1\hbox{ and } l=0,1,\ldots,p-1`, where :math:`x_{rqs}` are the complex data to
be transformed, :math:`\tilde{x}_{jkl}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.
