#include "ModelFail.hpp"

#include <iostream>

int ModelFail::getNumActions() const {
	return 2;
}

int ModelFail::getNumStates() const {
	return 1;
}

int ModelFail::getMaxTrajLen() const {
	return 2;
}

void ModelFail::generateTrajectories(vector<Trajectory> & buff, 
	const Policy & pi, 
	int numTraj, 
	mt19937_64 & generator) const 
{
	buff.resize(numTraj);
	normal_distribution<double> distribution(0,100);
	for (int trajCount = 0; trajCount < numTraj; trajCount++) {
		buff[trajCount].len = 0;
		buff[trajCount].actionProbabilities.resize(0);
		buff[trajCount].actions.resize(0);
		buff[trajCount].rewards.resize(0);
		buff[trajCount].states.resize(1);
		buff[trajCount].R = 0;
		int s = 0;
		buff[trajCount].states[0] = 0;
		for (int t = 0; t < getMaxTrajLen(); t++) {
			buff[trajCount].len++; // We have one more transition!
			// Get action
			int a = pi.getAction(0, generator);
			buff[trajCount].actions.push_back(a);
			double actionProbability = pi.getActionProbability(0, a);
			buff[trajCount].actionProbabilities.push_back(actionProbability);

			if (t == 0) {
				s = a;
			}

			// Update the reward
			double reward = 0;
			if (t == 1)
			{
				if (s == 0)
					reward = 1;
				else
					reward = -1;
			}
			
			buff[trajCount].rewards.push_back(reward);
			buff[trajCount].R += reward;

			if (t == 1)
				break;	// Entered a terminal state. Last transition

			// Add the state and features for the next element
			buff[trajCount].states.push_back(0);
		}
	}
}

double ModelFail::evaluatePolicy(
	const Policy & pi,
	mt19937_64 & generator) const 
{
	int numSamples = 10000;

	double result = 0;
	for (int trajCount = 0; trajCount < numSamples; trajCount++) {
		int s = 0;
		for (int t = 0; t < 2; t++) {
			int a = pi.getAction(0, generator);
			if (t == 0) {
				s = a;
			}

			// Update the reward
			double reward = 0;
			if (t == 1)
			{
				if (s == 0)
					reward = 1;
				else
					reward = -1;
			}
			result += reward;
		}
	}
	return result / (double)numSamples;
}

Policy ModelFail::getPolicy(int index) const {
	if (index == 1)
		return Policy("p1_ModelFail.txt", getNumActions(), getNumStates());
	if (index == 2)
		return Policy("p2_ModelFail.txt", getNumActions(), getNumStates());
	cerr << "Unknown policy index in ModelFail::getPolicy." << endl;
	exit(1);
}
