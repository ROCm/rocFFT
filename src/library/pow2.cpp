
#include <vector>
#include <assert.h>
#include <iostream>

#include "rocfft.h"
#include "./plan.h"
#include "./repo.h"
#include "./transform.h"
#include "./devicecall.h"

void PlanPow2(ExecPlan &execPlan)
{
	if(execPlan.execSeq.size() == 1)
	{
		if( 	(execPlan.execSeq[0]->precision == rocfft_precision_single) &&
			(execPlan.execSeq[0]->placement == rocfft_placement_inplace) &&
			(execPlan.execSeq[0]->inArrayType == rocfft_array_type_complex_interleaved) &&
			(execPlan.execSeq[0]->outArrayType == rocfft_array_type_complex_interleaved) &&
			(execPlan.execSeq[0]->dimension == 1) &&
			(execPlan.execSeq[0]->length[0] <= 4096) )
		{
			size_t WGS = 64;
			size_t NT = 1;
			DevFnCall ptr = nullptr;
	
			switch(execPlan.execSeq[0]->length[0])
			{
			case 4096: WGS = 256; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_4096); break;
			case 2048: WGS = 256; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2048); break;
			case 1024: WGS = 128; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1024); break;
			case 512:  WGS = 64;  NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_512); break;
			case 256:  WGS = 64;  NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_256); break;
			case 128:  WGS = 64;  NT = 4; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_128); break;
			case 64:   WGS = 64;  NT = 4; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_64); break;
			case 32:   WGS = 64; NT = 16; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_32); break;
			case 16:   WGS = 64; NT = 16; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_16); break;
			case 8:	   WGS = 64; NT = 32; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_8); break;
			case 4:	   WGS = 64; NT = 32; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_4); break;
			case 2:	   WGS = 64; NT = 64; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_2); break;
			case 1:	   WGS = 64; NT = 64; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1); break;
			}
	
			execPlan.devFnCall.push_back(ptr);
			GridParam gp;
			size_t B = execPlan.execSeq[0]->batch;
			gp.b_x = (B%NT) ? 1 + (B / NT) : (B / NT);
			gp.tpb_x = WGS;
			execPlan.gridParam.push_back(gp);
		}
	}
}

void TransformPow2(const ExecPlan &execPlan, void *in_buffer[], void *out_buffer[])
{
	DeviceCallIn data;
	DeviceCallOut back;

	data.node = execPlan.execSeq[0];
	data.bufIn = in_buffer[0];
	data.gridParam = execPlan.gridParam[0];

	DevFnCall fn = execPlan.devFnCall[0];
	fn(&data, &back);
}


