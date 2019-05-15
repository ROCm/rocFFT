/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
