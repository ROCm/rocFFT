
#ifndef REPO_H
#define REPO_H

#include <map>

extern "C"
{
typedef void (*DevFnCall)(void *, void *);
}

struct GridParam
{
	size_t b_x, b_y, b_z;
	size_t tpb_x, tpb_y, tpb_z;

	GridParam() : b_x(1), b_y(1), b_z(1), tpb_x(1), tpb_y(1), tpb_z(1)
	{}
};

struct ExecPlan
{
	TreeNode *rootPlan;
	std::vector<TreeNode *> execSeq;
	std::vector<DevFnCall> devFnCall;
	std::vector<GridParam> gridParam;
	size_t workBufSize;

	ExecPlan() : rootPlan(nullptr), workBufSize(0)
	{}
};

class Repo
{
	Repo() {}
	std::map<rocfft_plan_t, ExecPlan> planUnique;
	std::map<rocfft_plan, ExecPlan> execLookup;

public:
	Repo(const Repo &) = delete;
	Repo &operator=(const Repo &) = delete;

	static Repo &GetRepo()
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

	void CreatePlan(rocfft_plan plan);
	void GetPlan(rocfft_plan plan, ExecPlan &execPlan);
	void DeletePlan(rocfft_plan plan);
	
};

void ProcessNode(ExecPlan &execPlan);
void PrintNode(ExecPlan &execPlan);

void PlanPow2(ExecPlan &execPlan);

#endif // REPO_H

