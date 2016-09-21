
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
	void *bufIn;
	void *bufOut;

	GridParam gridParam;
};

struct DeviceCallOut
{
	int err;
};

extern "C"
{

void FN_PRFX(dfn_sp_ip_ci_ci_stoc_4096)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2048)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1024)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_512)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_256)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_128)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_64)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_32)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_16)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_8)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_4)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_2)(void *data_p, void *back_p);
void FN_PRFX(dfn_sp_ip_ci_ci_stoc_1)(void *data_p, void *back_p);

}



#endif // DEVICE_CALL_H

