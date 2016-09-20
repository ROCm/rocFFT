
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
};

struct DeviceCallOut
{
	int err;
};

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

void FN_PRFX(dfn_sp_ip_ci_ci_stoc_16)(void *data_p, void *back_p);

#ifdef __cplusplus
}
#endif // __cplusplus


#endif // DEVICE_CALL_H

