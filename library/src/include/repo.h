
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

#ifndef REPO_H
#define REPO_H

#include "plan.h"
#include "tree_node.h"
#include <map>
#include <mutex>

class Repo
{
    Repo() {}

    // keys to the maps are plan, deviceId.  plans must be per-device
    // since we allocate twiddles in the plan's TreeNode
    typedef std::pair<rocfft_plan_t, int> plan_unique_key_t;
    typedef std::pair<rocfft_plan, int>   exec_lookup_key_t;

    // planUnique has unique rocfft_plan_t and ExecPlan, and a reference counter
    std::map<plan_unique_key_t, std::pair<ExecPlan, int>> planUnique;
    std::map<exec_lookup_key_t, ExecPlan>                 execLookup;
    static std::mutex                                     mtx;

public:
    Repo(const Repo&) = delete; // delete is a c++11 feature, prohibit copy constructor
    Repo& operator=(const Repo&) = delete; // prohibit assignment operator

    static Repo& GetRepo()
    {
        static Repo repo;
        return repo;
    }

    ~Repo()
    {
        repoDestroyed = true;
    }

    static void CreatePlan(rocfft_plan plan);
    // may return nullptr if the plan is not known to the repo
    static ExecPlan* GetPlan(rocfft_plan plan);
    static void      DeletePlan(rocfft_plan plan);
    static size_t    GetUniquePlanCount();
    static size_t    GetTotalPlanCount();
    // remove cached plans
    static void Clear();

    // Repo is a singleton that should only be destroyed on static
    // deinitialization.  But it's possible for other things to want to
    // destroy plans at static deinitialization time.  So keep track of
    // whether the repo has been destroyed, so we can avoid wanting it
    // again.
    static std::atomic<bool> repoDestroyed;
};

#endif // REPO_H
