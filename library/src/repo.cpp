/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include <assert.h>
#include <iostream>
#include <vector>

#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "rocfft.h"

// Implementation of Class Repo

void Repo::CreatePlan(rocfft_plan plan)
{
    Repo& repo = Repo::GetRepo();
    // see if the repo has already stored the plan or not
    std::map<rocfft_plan_t, ExecPlan>::const_iterator it = repo.planUnique.find(*plan);
    if(it == repo.planUnique.end()) // if not found
    {
        TreeNode* rootPlan = TreeNode::CreateNode();

        rootPlan->dimension = plan->rank;
        rootPlan->batch     = plan->batch;
        for(size_t i = 0; i < plan->rank; i++)
        {
            rootPlan->length.push_back(plan->lengths[i]);

            rootPlan->inStride.push_back(plan->desc.inStrides[i]);
            rootPlan->outStride.push_back(plan->desc.outStrides[i]);
        }
        rootPlan->iDist = plan->desc.inDist;
        rootPlan->oDist = plan->desc.outDist;

        rootPlan->placement = plan->placement;
        rootPlan->precision = plan->precision;
        if((plan->transformType == rocfft_transform_type_complex_forward)
           || (plan->transformType == rocfft_transform_type_real_forward))
            rootPlan->direction = -1;
        else
            rootPlan->direction = 1;

        rootPlan->inArrayType  = plan->desc.inArrayType;
        rootPlan->outArrayType = plan->desc.outArrayType;

        ExecPlan execPlan;
        execPlan.rootPlan = rootPlan;
        ProcessNode(execPlan); // TODO: more descriptions are needed
        if(LOG_TRACE_ENABLED())
            PrintNode(*LogSingleton::GetInstance().GetTraceOS(), execPlan);

        PlanPowX(execPlan); // PlanPowX enqueues the GPU kernels by function
        // pointers but does not execute kernels
        repo.planUnique[*plan] = execPlan; // add this plan into member planUnique (type of map)
        repo.execLookup[plan]  = execPlan; // add this plan into member execLookup (type of map)
    }
    else // find the stored plan
    {
        repo.execLookup[plan] = it->second; // retrieve this plan and put it into member execLookup
    }
}
// according to input plan, return the corresponding execPlan
void Repo::GetPlan(rocfft_plan plan, ExecPlan& execPlan)
{
    Repo& repo = Repo::GetRepo();

    if(repo.execLookup.find(plan) != repo.execLookup.end())
        execPlan = repo.execLookup[plan];
}

void Repo::DeletePlan(rocfft_plan plan)
{
    Repo&                                     repo = Repo::GetRepo();
    std::map<rocfft_plan, ExecPlan>::iterator it   = repo.execLookup.find(plan);
    if(it != repo.execLookup.end())
        repo.execLookup.erase(it);
}
