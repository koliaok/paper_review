#pragma once

#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Policy {
public:
	Policy(const char * fileName, int numActions, int numStates);
	int getAction(const int & state, mt19937_64 & generator) const;
	double getActionProbability(const int & state, const int & action) const;
	VectorXd getActionProbabilities(const int & state) const;

private:
	int numActions;
	int numStates;
	VectorXd theta;

	static int wrand(mt19937_64 & generator, const VectorXd & probabilities);
};