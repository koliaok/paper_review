#include "Policy.h"
#include <fstream>

using namespace std;

Policy::Policy(const char * fileName, int numActions, int numStates) {
	this->numActions = numActions;
	this->numStates = numStates;
	theta.resize(numActions*numStates);
	ifstream in(fileName);
	for (int i = 0; i < (int)theta.size(); i++)
		in >> theta[i];
}

int Policy::getAction(const int & state, mt19937_64 & generator) const {
	return wrand(generator, getActionProbabilities(state));
}

double Policy::getActionProbability(const int & state, const int & action) const {
	return getActionProbabilities(state)(action);
}

VectorXd Policy::getActionProbabilities(const int & state) const {
	VectorXd actionProbabilities(numActions);
	for (int a = 0; a < numActions; a++)
		actionProbabilities[a] = theta[a*numStates + state];
	actionProbabilities.array() = actionProbabilities.array().exp();
	return actionProbabilities / actionProbabilities.sum();
}


int Policy::wrand(mt19937_64 & generator, const VectorXd & probabilities)
{
	double sum = 0;
	uniform_real_distribution<double> d(0, 1);
	double r = d(generator);
	for (int i = 0; i < (int)probabilities.size(); i++) {
		sum += (double)probabilities(i);
		if (sum >= r) return i;
	}
	// If we get here, there was a rounding error... doh. Pick a random action
	uniform_int_distribution<int> distribution(0, (int)probabilities.size() - 1);
	return distribution(generator);
}