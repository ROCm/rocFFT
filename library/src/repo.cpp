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
#include "node_factory.h"
#include "plan.h"
#include "repo.h"
#include "rocfft.h"

// Implementation of Class Repo

std::mutex        Repo::mtx;
std::atomic<bool> Repo::repoDestroyed(false);

rocfft_status Repo::CreatePlan(rocfft_plan plan)
{
    if(plan == nullptr)
        return rocfft_status_failure;

    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed)
        return rocfft_status_failure;

    Repo& repo = Repo::GetRepo();

    // see if the repo has already stored the plan or not
    int deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
        return rocfft_status_failure;
    plan_unique_key_t uniqueKey{*plan, deviceId};
    exec_lookup_key_t lookupKey{plan, deviceId};

    auto it = repo.planUnique.find(uniqueKey);
    if(it == repo.planUnique.end()) // if not found
    {
        NodeMetaData rootPlanData(nullptr);

        rootPlanData.dimension = plan->rank;
        rootPlanData.batch     = plan->batch;
        for(size_t i = 0; i < plan->rank; i++)
        {
            rootPlanData.length.push_back(plan->lengths[i]);

            rootPlanData.inStride.push_back(plan->desc.inStrides[i]);
            rootPlanData.outStride.push_back(plan->desc.outStrides[i]);
        }
        rootPlanData.iDist = plan->desc.inDist;
        rootPlanData.oDist = plan->desc.outDist;

        rootPlanData.placement = plan->placement;
        rootPlanData.precision = plan->precision;
        if((plan->transformType == rocfft_transform_type_complex_forward)
           || (plan->transformType == rocfft_transform_type_real_forward))
            rootPlanData.direction = -1;
        else
            rootPlanData.direction = 1;

        rootPlanData.inArrayType  = plan->desc.inArrayType;
        rootPlanData.outArrayType = plan->desc.outArrayType;
        rootPlanData.rootIsC2C    = (rootPlanData.inArrayType != rocfft_array_type_real)
                                 && (rootPlanData.outArrayType != rocfft_array_type_real);

        ExecPlan execPlan;
        execPlan.rootPlan = NodeFactory::CreateExplicitNode(rootPlanData, nullptr);

        std::copy(plan->lengths.begin(),
                  plan->lengths.begin() + plan->rank,
                  std::back_inserter(execPlan.iLength));
        std::copy(plan->lengths.begin(),
                  plan->lengths.begin() + plan->rank,
                  std::back_inserter(execPlan.oLength));

        if(plan->transformType == rocfft_transform_type_real_inverse)
        {
            execPlan.iLength.front() = execPlan.iLength.front() / 2 + 1;
            if(plan->placement == rocfft_placement_inplace)
                execPlan.oLength.front() = execPlan.iLength.front() * 2;
        }
        if(plan->transformType == rocfft_transform_type_real_forward)
        {
            execPlan.oLength.front() = execPlan.oLength.front() / 2 + 1;
            if(plan->placement == rocfft_placement_inplace)
                execPlan.iLength.front() = execPlan.oLength.front() * 2;
        }

        if(hipGetDeviceProperties(&(execPlan.deviceProp), deviceId) != hipSuccess)
            return rocfft_status_failure;

        try
        {
            ProcessNode(execPlan); // TODO: more descriptions are needed
        }
        catch(std::exception& e)
        {
            rocfft_cerr << e.what() << std::endl;
            if(LOG_PLAN_ENABLED())
                PrintNode(*LogSingleton::GetInstance().GetPlanOS(), execPlan);
            return rocfft_status_failure;
        }

        if(!PlanPowX(execPlan)) // PlanPowX enqueues the GPU kernels by function
        {
            return rocfft_status_failure;
        }

        // pointers but does not execute kernels

        // add this plan into member planUnique (type of map)
        repo.planUnique[{*plan, deviceId}] = std::make_pair(execPlan, 1);
        // add this plan into member execLookup (type of map)
        repo.execLookup[lookupKey] = execPlan;
    }
    else // find the stored plan
    {
        repo.execLookup[lookupKey]
            = it->second.first; // retrieve this plan and put it into member execLookup
        it->second.second++;
    }

    return rocfft_status_success;
}
// According to input plan, return the corresponding execPlan
ExecPlan* Repo::GetPlan(rocfft_plan plan)
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed || plan == nullptr)
        return nullptr;

    Repo& repo     = Repo::GetRepo();
    int   deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
        return nullptr;
    exec_lookup_key_t lookupKey{plan, deviceId};

    auto it = repo.execLookup.find(lookupKey);
    if(it != repo.execLookup.end())
        return &it->second;
    return nullptr;
}

// Remove the plan from Repo and release its ExecPlan resources if it is the last reference
void Repo::DeletePlan(rocfft_plan plan)
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed || plan == nullptr)
        return;

    Repo& repo     = Repo::GetRepo();
    int   deviceId = 0;
    if(hipGetDevice(&deviceId) != hipSuccess)
        return;
    exec_lookup_key_t lookupKey{plan, deviceId};
    plan_unique_key_t uniqueKey{*plan, deviceId};

    auto it = repo.execLookup.find(lookupKey);
    if(it != repo.execLookup.end())
    {
        repo.execLookup.erase(it);
    }

    auto it_u = repo.planUnique.find(uniqueKey);
    if(it_u != repo.planUnique.end())
    {
        it_u->second.second--;
        if(it_u->second.second <= 0)
        {
            repo.planUnique.erase(it_u);
        }
    }
}

size_t Repo::GetUniquePlanCount()
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed)
        return 0;

    Repo& repo = Repo::GetRepo();
    return repo.planUnique.size();
}

size_t Repo::GetTotalPlanCount()
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed)
        return 0;

    Repo& repo = Repo::GetRepo();
    return repo.execLookup.size();
}

void Repo::Clear()
{
    std::lock_guard<std::mutex> lck(mtx);
    if(repoDestroyed)
        return;
    Repo& repo = Repo::GetRepo();
    repo.planUnique.clear();
    repo.execLookup.clear();
}
