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

std::mutex        Repo::mtx;
std::atomic<bool> Repo::repoDestroyed(false);

// for callbacks, work out which nodes of the plan are loading data
// from global memory, and storing data to global memory
static std::pair<TreeNode*, TreeNode*> get_load_store_nodes(const ExecPlan& execPlan)
{
    const auto& seq = execPlan.execSeq;

    // look forward for the first node that reads from input
    auto      load_it = std::find_if(seq.begin(), seq.end(), [&](const TreeNode* n) {
        return n->obIn == execPlan.rootPlan->obIn;
    });
    TreeNode* load    = load_it == seq.end() ? nullptr : *load_it;

    // look backward for the last node that writes to output
    auto      store_it = std::find_if(seq.rbegin(), seq.rend(), [&](const TreeNode* n) {
        return n->obOut == execPlan.rootPlan->obOut;
    });
    TreeNode* store    = store_it == seq.rend() ? nullptr : *store_it;

    assert(load && store);
    return std::make_pair(load, store);
}

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
        auto rootPlan = TreeNode::CreateNode();

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

        // assign callbacks to root plan, so plan building can know about them
        rootPlan->callbacks = plan->desc.callbacks;

        ExecPlan execPlan;
        execPlan.rootPlan = std::move(rootPlan);
        if(hipGetDeviceProperties(&(execPlan.deviceProp), deviceId) != hipSuccess)
            return rocfft_status_failure;

        ProcessNode(execPlan); // TODO: more descriptions are needed
        if(LOG_TRACE_ENABLED())
            PrintNode(*LogSingleton::GetInstance().GetTraceOS(), execPlan);

        if(!PlanPowX(execPlan)) // PlanPowX enqueues the GPU kernels by function
        {
            return rocfft_status_failure;
        }

        TreeNode* load_node             = nullptr;
        TreeNode* store_node            = nullptr;
        std::tie(load_node, store_node) = get_load_store_nodes(execPlan);

        // now that plan building is finished, assign callbacks to
        // the leaf nodes that are actually doing the work
        load_node->callbacks.load_cb_fn        = plan->desc.callbacks.load_cb_fn;
        load_node->callbacks.load_cb_data      = plan->desc.callbacks.load_cb_data;
        load_node->callbacks.load_cb_lds_bytes = plan->desc.callbacks.load_cb_lds_bytes;

        store_node->callbacks.store_cb_fn        = plan->desc.callbacks.store_cb_fn;
        store_node->callbacks.store_cb_data      = plan->desc.callbacks.store_cb_data;
        store_node->callbacks.store_cb_lds_bytes = plan->desc.callbacks.store_cb_lds_bytes;

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
