# rocFFT API design


## Summary

In this document, I attempt to explain the rationale behind the design of rocFFT API. In designing the rocFFT API, I studied other popular FFT libraries to understand and compare their interfaces and usage. These include FFTW, Intel MKL FFT, Nvidia cuFFT, and our own clFFT. At the beginning, my desire was to create an interface closely resembling FFTW given its wide adoption. But after writing some preliminary interface code, I dropped that idea for a number of reasons. I should note though, that the overall usage structure (with 2 distinct stages: an initial plan definition stage, and a subsequent execution stage) still remain the same in rocFFT, similar to clFFT and rooted in FFTW. With the current design of rocFFT, my approach has been to take clFFT's API and substantially improve it based on user experience and feedback. At the same time, I am making the usage as simple as possible and intuitive for the common cases. Some of the main reasons for my approach are:

 * FFTW duplicates every function for each FP precision, this is cumbersome
 * it is designed for CPU, doesn't take into account GPU programming constraints
 * three levels of planning interfaces (basic, advanced, guru) may be good to target a wide variety of users; not necessary given our goals
 * it is always possible to provide an exact drop-in replacement FFTW API as an additional/separate interface in rocFFT in the future
 * similarity with clFFT API helps with continuity for current OpenCL users looking to switch to ROC
 * C API is only the first step; we will define a C++ interface layer on top
 * clFFT's custom data layout specification falls at a level between the advanced and guru interface of FFTW; and is plenty powerful; no user has ever asked for guru level functionality

## Memory management

As in FFTW, rocFFT defines its own memory allocation and free functions. This helps the library to have control over memory it has to use. It is strongly recommended for users to allocate and release memory using these functions. These functions allocate/free memory space on the **device**.

```c
rocfft_status rocfft_buffer_create_with_alloc(  rocfft_buffer *buffer,
                                                rocfft_element_type element_type,
                                                size_t size_in_elements );
rocfft_status rocfft_buffer_destroy( rocfft_buffer buffer );
```

Here, 'buffer' parameter is a pointer to an internal library object that holds details about the device buffer. The parameter 'element_type' specifies the type of element/unit the buffer is made of. And 'size_in_elements' specifies the number of elements in the buffer.

Since the library interface accepts only buffers of type 'rocfft_buffer', the library provides couple of functions to work with memory created directly with underlying runtime API.

```c
rocfft_status rocfft_buffer_create_with_ptr(rocfft_buffer *buffer, void *p);
rocfft_status rocfft_buffer_get_ptr(rocfft_buffer buffer, void **p);
```

The function 'rocfft_buffer_create_with_ptr' is to create a rocfft_buffer object that simply uses device buffer already created and pointed by 'p'. Similarly it is possible to retrieve the raw pointer from a rocfft_buffer object using the function 'rocfft_buffer_get_ptr'. It is important to note that memory created by user must be freed by user. The library only frees memory (the user must call 'rocfft_buffer_destroy') that it created inside 'rocfft_buffer_create_with_alloc'.

## Plan definition

There is a single step (as opposed to 2 steps in clFFT) to create a plan object in rocFFT.

```c
rocfft_status rocfft_plan_create(       rocfft_plan *plan,
                                        rocfft_transform_type transform_type, rocfft_precision precision,
                                        size_t dimensions, const size_t *lengths, size_t number_of_transforms,
                                        const rocfft_description *description );
```

Here, 'plan' parameter is a pointer to an internal object created by library that holds plan information. The parameters 'transform_type' and 'precision' specify the fundamental type and precision of the transform. 'dimensions' specify the number of dimensions in the data. Valid values are 1, 2 and 3. The 'lengths' array specifies size in each dimension. Unless custom strides are specified, the data is assumed to be packed. It is important to note that lengths[0] specifies the size of the dimension where consecutive elements are contiguous in memory. The lengths[1], if applicable, is the next higher dimension and so on. The 'number_of_transforms' parameter specifies how many transforms (of the same kind) needs to be computed. By specifying a value greater than 1, an array of transforms can be computed. The 'description' parameter can be set to NULL if no further specification is necessary. Or a description object, set up using other api functions, can be passed in to specify more plan properties.


By default results are written back to the input buffer, an in-place result placement. To specify not in-place result placement, the following function can be used to set up the description object to be passed subsequently to 'rocfft_plan_create'. This function can be used to specify input and output array types. Not all combinations of array types are supported and error code will be returned for unsupported cases. Additionally, input and output buffer offsets can be specified using this function.

```c
rocfft_status rocfft_description_set_data_outline(      rocfft_description *description,
                                                        rocfft_result_placement placement,
                                                        rocfft_array_type in_array_type, rocfft_array_type out_array_type,
                                                        const size_t *in_offsets, const size_t *out_offsets );
```

The following function can be used to specify custom layout of data, with the ability to specify stride between consecutive elements in all dimensions. Also, distance between transform array members can be specified, and they take meaning if the 'number_of_transforms' parameter in 'rocfft_plan_create' is greater than 1.

```c
rocfft_status rocfft_description_set_data_layout(       rocfft_description *description,
                                                        const size_t *in_strides, size_t in_distance,
                                                        const size_t *out_strides, size_t out_distance );

```

The following function can be used to change the default device or add a set of devices for which the plan has to be created.

```c
rocfft_status rocfft_description_set_devices(   rocfft_description description,
                                                void *devices,
                                                size_t number_of_devices );
```

To destruct a plan after it is no longer needed, the following function can be used.

```c
rocfft_status rocfft_plan_destroy( rocfft_plan plan );
```

## Execution of plan
After a plan is created, the library can be instructed to execute that plan on input/output data using the function shown below. If the transform is in-place, only the input buffer is needed and the output buffer parameter can be set to NULL. For not in-place transforms, output buffers have to be specified. The final parameter in this function is an execution_info object. This parameter serves as both a way for the user to control execution related things, as well as for the library to pass any information back to the user.

```c
rocfft_status rocfft_execute(   const rocfft_plan plan,
                                rocfft_buffer *in_buffer,
                                rocfft_buffer *out_buffer,
                                rocfft_execution_info info );
```

The following functions can be used to create and destroy execution_info objects.

```c
rocfft_status rocfft_execution_info_create(rocfft_execution_info *info);
rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);
```
					
As an example of the usage of execution_info object, consider the following functions.

```c
rocfft_status rocfft_execution_info_set_mode( rocfft_execution_info info, rocfft_execution_mode mode );
rocfft_status rocfft_execution_info_set_work_buffer( rocfft_execution_info info, rocfft_buffer work_buffer );
rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, void *stream);

rocfft_status rocfft_execution_info_get_events( const rocfft_execution_info info,
                                                void **events,
                                                size_t number_of_events);
```

In the function 'rocfft_execution_info_set_mode' shown above, the execution_info object is used to control the execution mode. Appropriate enumeration value can be specified to control blocking/non-blocking behavior. It serves as an input to the library and has to be called before a call to the rocfft_execute function. This is applicable to all of the **set** functions shown above. The function 'rocfft_execution_info_set_work_buffer' can be used to pass buffers created by the user to the library if for any reason user does not prefer library allocating/freeing device memory from inside 'rocfft_execute' function. The function 'rocfft_execution_info_set_stream' can be used to set the underlying device queue/stream where the library computations would be inserted. The library assumes user has created such a stream in the program and merely assigns work to the stream. The function 'rocfft_execution_info_get_events' can be used to get handles to events the library created around one or more kernel launches inside the library. Needless to say, this function and other **get** functions are called after a call to 'rocfft_execute'. 

## Enumeration types and values
Documentation is TBD.

```c

// Status & error message
typedef enum rocfft_status_e
{
	rocfft_status_success,
	rocfft_status_failure,
} rocfft_status;

// Type of transform
typedef enum rocfft_transform_type_e
{
	rocfft_transform_type_complex_forward,
	rocfft_transform_type_complex_inverse,
	rocfft_transform_type_real_forward,
	rocfft_transform_type_real_inverse,	
} rocfft_transform_type;

// Precision
typedef enum rocfft_precision_e
{
	rocfft_precision_single,
	rocfft_precision_double,
} rocfft_precision;

// Element type
typedef enum rocfft_element_type_e
{
	rocfft_element_type_complex_single,
	rocfft_element_type_complex_double,
	rocfft_element_type_single,
	rocfft_element_type_double,
	rocfft_element_type_byte,	
} rocfft_element_type;

// Result placement
typedef enum rocfft_result_placement_e
{
	rocfft_placement_inplace,
	rocfft_placement_notinplace,	
} rocfft_result_placement;

// Array type
typedef enum rocfft_array_type_e
{
	rocfft_array_type_complex_interleaved,
	rocfft_array_type_complex_planar,
	rocfft_array_type_real,
	rocfft_array_type_hermitian_interleaved,
	rocfft_array_type_hermitian_planar,	
} rocfft_array_type;

// Execution mode
typedef enum rocfft_execution_mode_e
{
	rocfft_exec_mode_nonblocking,
	rocfft_exec_mode_nonblocking_with_flush,
	rocfft_exec_mode_blocking,
} rocfft_execution_mode;

```

## Usage of the API
To give an idea of how the library API is intended to be used, the following sequence of function calls and pseudo-code is provided.

```c

// allocate buffers
status = rocfft_buffer_create_with_alloc(&buffer_a, ...);
status = rocfft_buffer_create_with_alloc(&buffer_b, ...);

// get pointers to enable read/write of data
status = rocfft_buffer_get_ptr(buffer_a, &raw_ptr_a);
status = rocfft_buffer_get_ptr(buffer_a, &raw_ptr_b);

// initialize input
...

// setup description if needed
rocfft_description description = NULL;
status = rocfft_description_set_data_outline(&description, rocfft_placement_notinplace, ...);

// create plan
status = rocfft_plan_create(&plan, ..., &description);

// create execution_info as needed
status = rocfft_execution_info_create(&execution_info);
status = rocfft_execution_info_set_mode(execution_info, rocfft_exec_mode_blocking);

// execute the plan
status = rocfft_execute(plan, &buffer_a, &buffer_b, execution_info);

// analyze results
...

// destruct library objects
status = rocfft_execution_info_destroy(execution_info);

// destruct plan
status = rocfft_plan_destroy(plan);

// destruct buffers
status = rocfft_buffer_destroy(buffer_a);
status = rocfft_buffer_destroy(buffer_b);
```
