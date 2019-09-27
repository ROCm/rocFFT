/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <assert.h>
#include <iostream>
#include <vector>

#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "rocfft.h"

// Implementation of Class Repo

std::mutex Repo::mtx;

void Repo::CreatePlan(rocfft_plan plan)
{
    Repo&                       repo = Repo::GetRepo();
    std::lock_guard<std::mutex> lck(mtx);

    // see if the repo has already stored the plan or not
    auto it = repo.planUnique.find(*plan);
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
        repo.planUnique[*plan] = std::pair<ExecPlan, int>(
            execPlan, 1); // add this plan into member planUnique (type of map)
        repo.execLookup[plan] = execPlan; // add this plan into member execLookup (type of map)
    }
    else // find the stored plan
    {
        repo.execLookup[plan]
            = it->second.first; // retrieve this plan and put it into member execLookup
        it->second.second++;
    }
}
// According to input plan, return the corresponding execPlan
void Repo::GetPlan(rocfft_plan plan, ExecPlan& execPlan)
{
    Repo&                       repo = Repo::GetRepo();
    std::lock_guard<std::mutex> lck(mtx);
    if(repo.execLookup.find(plan) != repo.execLookup.end())
        execPlan = repo.execLookup[plan];
}

// Remove the plan from Repo and release its ExecPlan resources if it is the last reference
void Repo::DeletePlan(rocfft_plan plan)
{
    Repo&                       repo = Repo::GetRepo();
    std::lock_guard<std::mutex> lck(mtx);
    auto                        it = repo.execLookup.find(plan);
    if(it != repo.execLookup.end())
    {
        repo.execLookup.erase(it);
    }

    auto it_u = repo.planUnique.find(*plan);
    if(it_u != repo.planUnique.end())
    {
        it_u->second.second--;
        if(it_u->second.second <= 0)
        {
            TreeNode::DeleteNode(it_u->second.first.rootPlan);
            it_u->second.first.rootPlan = nullptr;
            repo.planUnique.erase(it_u);
        }
    }
}

size_t Repo::GetUniquePlanCount()
{
    Repo&                       repo = Repo::GetRepo();
    std::lock_guard<std::mutex> lck(mtx);
    return repo.planUnique.size();
}

size_t Repo::GetTotalPlanCount()
{
    Repo&                       repo = Repo::GetRepo();
    std::lock_guard<std::mutex> lck(mtx);
    return repo.execLookup.size();
}
