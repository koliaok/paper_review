#pragma once

#include "Trajectory.h"
#include <vector>
#include "Policy.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// This class builds a model of an MDPs with tabular states and actions, and a 
// finite  horizon, L.

class Model {
public:
	// Adding a defauly constructor so that I can extend it
	Model();

	// Take historical data in trajs and build a model. numState and numActions
	// are the total numbers of possible states and actions
	Model(const vector<Trajectory*> & trajs, int numStates, int numActions, int L, bool JiangStyle);

	// V and Q predictions - can be loaded for a specific evaluation policy
	// int L = Gridworld::getMaxTrajLen();
	void loadEvalPolicy(const Policy & pi, const int & L);

	vector<VectorXd> actionProbabilities; // [s][a]
	vector<VectorXd> V; // [t](s) - t in [0,L].
	vector<MatrixXd> Q; // [t](s,a)
	double evalPolicyValue;

	// Generate trajectories from the provided policy
	vector<Trajectory> generateTrajectories(const Policy & pi, int N, mt19937_64 & generator) const;

	// R[s][a][s']. Size = [numStates][numActions][numStates+1]
	vector<vector<vector<double>>> R;

	// P[s][a][s']. Size = [numStates][numActions][numStates+1]
	vector<vector<vector<double>>> P;

private:
	int N;
	int L;
	int numStates;
	int numActions;
	vector<double> d0;	// d0[s] = Pr(S_0=s). Size = [numStates]

	// How many times was each (s), (s,a), and (s,a,s') tuple seen?
	vector<vector<int>> stateActionCounts;

	// Includes transitions to terminal absorbing state due to time horizon
	vector<vector<int>> stateActionCounts_includingHorizon;

	vector<vector<vector<int>>> stateActionStateCounts;

	// Includes transitions to terminal absorbing state due to time horizon
	vector<vector<vector<int>>> stateActionStateCounts_includingHorizon;

	template <typename VectorType>
	static int wrand(mt19937_64 & generator, const VectorType & probabilities)
	{
		double sum = 0;
		uniform_real_distribution<double> d(0, 1);
		double r = d(generator);
		for (int i = 0; i < (int)probabilities.size(); i++) {
			sum += (double)probabilities[i];
			if (sum >= r) return i;
		}
		// If we get here, there was a rounding error. Pick a random action
		uniform_int_distribution<int> d2(0, (int)probabilities.size() - 1);
		return d2(generator);
	}
};