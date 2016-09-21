
#ifndef TRANSFORM_H
#define TRANSFORM_H

struct rocfft_execution_info_t
{
	void *workBuffer;
	size_t workBufferSize;
};

void TransformPow2(const ExecPlan &execPlan, void *in_buffer[], void *out_buffer[]);

#endif // TRANSFORM_H

