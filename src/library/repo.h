
#ifndef REPO_H
#define REPO_H

#include <map>

struct ExecPlan
{
	TreeNode *rootPlan;
	std::vector<TreeNode *> execSeq;
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

#endif // REPO_H

