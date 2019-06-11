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

#include "tree_node.h"
#include <map>

class Repo
{
    Repo() {}
    std::map<rocfft_plan_t, ExecPlan> planUnique;
    std::map<rocfft_plan, ExecPlan>   execLookup;

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
        std::map<rocfft_plan_t, ExecPlan>::iterator it = planUnique.begin();
        while(it != planUnique.end())
        {
            TreeNode::DeleteNode(it->second.rootPlan);
            it->second.rootPlan = nullptr;
            it++;
        }
    }

    static void CreatePlan(rocfft_plan plan);
    static void GetPlan(rocfft_plan plan, ExecPlan& execPlan);
    static void DeletePlan(rocfft_plan plan);
};

#endif // REPO_H
