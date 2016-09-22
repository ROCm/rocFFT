
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
	for(size_t i=0; i<execPlan.execSeq.size(); i++)
	{
		if(	(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) ||
			(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) ||
			(execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) )
		{
			execPlan.execSeq[i]->twiddles = twiddles_create(execPlan.execSeq[i]->length[0]);				
		}

		if(execPlan.execSeq[i]->large1D != 0)
		{
			execPlan.execSeq[i]->twiddles_large = twiddles_create(execPlan.execSeq[i]->large1D);
		}
	}

	if(execPlan.rootPlan->dimension == 1)
	{
		if(execPlan.execSeq.size() == 1)
		{
			if( 	(execPlan.execSeq[0]->precision == rocfft_precision_single) &&
				(execPlan.execSeq[0]->placement == rocfft_placement_inplace) &&
				(execPlan.execSeq[0]->inArrayType == rocfft_array_type_complex_interleaved) &&
				(execPlan.execSeq[0]->outArrayType == rocfft_array_type_complex_interleaved) )
			{
				assert(execPlan.execSeq[0]->length[0] <= 4096);

				size_t WGS = 64;
				size_t NT = 1;
				DevFnCall ptr = nullptr;
	
				switch(execPlan.execSeq[0]->length[0])
				{
				case 4096: WGS = 256; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4096); break;
				case 2048: WGS = 256; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2048); break;
				case 1024: WGS = 128; NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1024); break;
				case 512:  WGS = 64;  NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_512); break;
				case 256:  WGS = 64;  NT = 1; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_256); break;
				case 128:  WGS = 64;  NT = 4; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_128); break;
				case 64:   WGS = 64;  NT = 4; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_64); break;
				case 32:   WGS = 64; NT = 16; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_32); break;
				case 16:   WGS = 64; NT = 16; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_16); break;
				case 8:	   WGS = 64; NT = 32; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_8); break;
				case 4:	   WGS = 64; NT = 32; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4); break;
				case 2:	   WGS = 64; NT = 64; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2); break;
				case 1:	   WGS = 64; NT = 64; ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_1); break;
				}
	
				execPlan.devFnCall.push_back(ptr);
				GridParam gp;
				size_t B = execPlan.execSeq[0]->batch;
				gp.b_x = (B%NT) ? 1 + (B / NT) : (B / NT);
				gp.tpb_x = WGS;
				execPlan.gridParam.push_back(gp);
			}
		}
		else
		{
			for(size_t i=0; i<execPlan.execSeq.size(); i++)
			{
				DevFnCall ptr = nullptr;
				GridParam gp;

				if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq.size() == 2) )
				{
					if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 128) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_128);
						gp.b_x = 8 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
					if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 256) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_256);
						gp.b_x = 16 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
					if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 256) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_128_256);
						gp.b_x = 32 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
					if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_256_256);
						gp.b_x = 32 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 256;
					}
				}
				else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC) && (execPlan.execSeq.size() == 2) )
				{
					if( (execPlan.execSeq[i]->length[0] == 128) && (execPlan.execSeq[i]->length[1] == 64) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_128_64);
						gp.b_x = 8 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
					if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 64) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_64);
						gp.b_x = 8 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 256;
					}
					if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 128) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_128);
						gp.b_x = 16 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 256;
					}
					if( (execPlan.execSeq[i]->length[0] == 256) && (execPlan.execSeq[i]->length[1] == 256) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbrc_2_256_256);
						gp.b_x = 32 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 256;
					}
				}
				else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC) && (execPlan.execSeq.size() == 3) )
				{
					if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 2048) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_2048);
						gp.b_x = 128 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
					if( (execPlan.execSeq[i]->length[0] == 64) && (execPlan.execSeq[i]->length[1] == 4096) )
					{
						ptr = FN_PRFX(dfn_sp_op_ci_ci_sbcc_2_64_4096);
						gp.b_x = 256 * execPlan.execSeq[i]->batch;
						gp.tpb_x = 128;
					}
				}
				else if( (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM) && (execPlan.execSeq.size() == 3) )
				{
					switch(execPlan.execSeq[i]->length[0])
					{
					case 4096: 	gp.tpb_x = 256; gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;
							ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_4096); break;
					case 2048: 	gp.tpb_x = 256; gp.b_x = execPlan.execSeq[i]->length[1] * execPlan.execSeq[i]->batch;
							ptr = &FN_PRFX(dfn_sp_ip_ci_ci_stoc_1_2048); break;
					default: assert(false);
					}
				}
				else if(execPlan.execSeq[i]->scheme == CS_KERNEL_TRANSPOSE)
				{
					ptr = FN_PRFX(transpose_var1);
					gp.tpb_x = 16;
					gp.tpb_y = 16;
					if(execPlan.execSeq[i]->transTileDir == TTD_IP_HOR)
					{
						gp.b_x = execPlan.execSeq[i]->length[0] / 64;
						gp.b_y = (execPlan.execSeq[i]->length[1] / 64) * execPlan.execSeq[i]->batch;
					}
					else
					{
						gp.b_x = execPlan.execSeq[i]->length[1] / 64;
						gp.b_y = (execPlan.execSeq[i]->length[0] / 64) * execPlan.execSeq[i]->batch;
					}
				}

				execPlan.devFnCall.push_back(ptr);
				execPlan.gridParam.push_back(gp);
			}
		}
	}

}

void TransformPow2(const ExecPlan &execPlan, void *in_buffer[], void *out_buffer[], rocfft_execution_info info)
{
	assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
	assert(execPlan.execSeq.size() == execPlan.gridParam.size());

	if(execPlan.rootPlan->dimension == 1)
	{
		if(execPlan.execSeq.size() == 1)
		{
			DeviceCallIn data;
			DeviceCallOut back;

			data.node = execPlan.execSeq[0];
			data.bufIn[0] = in_buffer[0];
			data.gridParam = execPlan.gridParam[0];

			DevFnCall fn = execPlan.devFnCall[0];
			fn(&data, &back);
		}
		else
		{
			for(size_t i=0; i<execPlan.execSeq.size(); i++)
			{
				DeviceCallIn data;
				DeviceCallOut back;

				data.node = execPlan.execSeq[i];

				switch(data.node->obIn)
				{
				case OB_USER_IN:	data.bufIn[0] = in_buffer[0]; break;
				case OB_USER_OUT:	data.bufIn[0] = out_buffer[0]; break;
				case OB_TEMP:		data.bufIn[0] = info->workBuffer; break;
				default: assert(false);
				}

				switch(data.node->obOut)
				{
				case OB_USER_IN:	data.bufOut[0] = in_buffer[0]; break;
				case OB_USER_OUT:	data.bufOut[0] = out_buffer[0]; break;
				case OB_TEMP:		data.bufOut[0] = info->workBuffer; break;
				default: assert(false);
				}

				data.gridParam = execPlan.gridParam[i];

				DevFnCall fn = execPlan.devFnCall[i];
				fn(&data, &back);
			}
		}
	}
}


