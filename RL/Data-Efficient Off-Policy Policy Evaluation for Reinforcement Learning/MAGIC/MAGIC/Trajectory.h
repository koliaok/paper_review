#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct Trajectory {
	int len;
	vector<int> states;
	vector<int> actions;
	vector<double> rewards;
	vector<double> actionProbabilities;
	double R;

	// These are loaded for a specific evaluation policy

	// Will be loaded with per-time-step importance weights. 
	// IWs[0] = pi_e(first action) / pi_b(first action)
	VectorXd IWs;	

	// Cumulative importance weights - loaded with product of importance 
	// weights up to current time. [0] = pi_e(first action)/pi_b(first action)
	VectorXd cumIWs;

	vector<double> evalActionProbabilities;
};