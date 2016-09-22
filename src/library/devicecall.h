
#ifndef DEVICE_CALL_H
#define DEVICE_CALL_H

#define FN_PRFX(X) rocfft_internal_ ## X

#include "rocfft.h"
#include "./plan.h"
#include "./repo.h"
#include "./transform.h"

struct DeviceCallIn
{
	TreeNode *node;
	void *bufIn[2];
	void *bufOut[2];

	GridParam gridParam;
};

struct DeviceCallOut
{
	int err;
};

extern "C"
{

void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4096)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2048)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1024)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_512)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_256)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_128)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_64)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_32)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_16)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_8)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1)(void *data_p, void *back_p);

void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_128)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_256)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_128_256)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_256_256)(void *data_p, void *back_p);

void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_2048)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_4096)(void *data_p, void *back_p);

void FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_128_64)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_64)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_128)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_256)(void *data_p, void *back_p);

void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_4096)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_2048)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_1024)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2_512)(void *data_p, void *back_p);

void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_4096)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_2048)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_1024)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_op_ci_ci_stoc_2_512)(void *data_p, void *back_p);

void FN_PRFX(transpose_var1)(void *data_p, void *back_p);

}



#endif // DEVICE_CALL_H

