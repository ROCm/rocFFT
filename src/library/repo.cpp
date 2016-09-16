#include <vector>
#include <assert.h>
#include <iostream>

#include "rocfft.h"
#include "./plan.h"
#include "./repo.h"

void Repo::CreatePlan(rocfft_plan plan)
{
	Repo &repo = Repo::GetRepo();
	if(repo.planUnique.find(*plan) == repo.planUnique.end())
	{
		TreeNode *rootPlan = TreeNode::CreateNode();

		rootPlan->dimension = plan->rank;
		rootPlan->batch = plan->batch;
		for(size_t i=0; i<plan->rank; i++)
		{
			rootPlan->length.push_back(plan->lengths[i]);
			
			rootPlan->inStride.push_back(plan->desc.inStrides[i]);
			rootPlan->outStride.push_back(plan->desc.outStrides[i]);
		}
		rootPlan->iDist = plan->desc.inDist;
		rootPlan->oDist = plan->desc.outDist;

		rootPlan->placement = plan->placement;
		rootPlan->precision = plan->precision;

		rootPlan->inArrayType  = plan->desc.inArrayType;
		rootPlan->outArrayType = plan->desc.outArrayType;

		ExecPlan execPlan;
		execPlan.rootPlan = rootPlan;
		ProcessNode(execPlan);
		planUnique[*plan] = execPlan;

		execLookup[plan] = execPlan;
	}
}

void Repo::GetPlan(rocfft_plan plan, ExecPlan &execPlan)
{
	Repo &repo = Repo::GetRepo();

	if(repo.execLookup.find(plan) != repo.execLookup.end())
		execPlan = repo.execLookup[plan];
}

void Repo::DeletePlan(rocfft_plan plan)
{
	Repo &repo = Repo::GetRepo();
	std::map<rocfft_plan, ExecPlan>::iterator it = repo.execLookup.find(plan);
	if(it != repo.execLookup.end())
		repo.execLookup.erase(it);
}



