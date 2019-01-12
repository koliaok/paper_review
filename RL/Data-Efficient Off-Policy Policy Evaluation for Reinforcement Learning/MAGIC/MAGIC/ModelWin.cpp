#include "ModelWin.hpp"

#include <iostream>

int ModelWin::getNumActions() const {
	return 2;
}

int ModelWin::getNumStates() const {
	return 3;
}

int ModelWin::getMaxTrajLen() const {
	return 20;
}

void ModelWin::generateTrajectories(vector<Trajectory> & buff, const Policy & pi, int numTraj, mt19937_64 & generator) const {
	buff.resize(numTraj);
	bernoulli_distribution p6true(0.6);
	for (int trajCount = 0; trajCount < numTraj; trajCount++) {
		buff[trajCount].len = 0;
		buff[trajCount].actionProbabilities.resize(0);
		buff[trajCount].actions.resize(0);
		buff[trajCount].rewards.resize(0);
		buff[trajCount].states.resize(1);
		buff[trajCount].R = 0;
		int s = 0;
		buff[trajCount].states[0] = s;
		for (int t = 0; t < getMaxTrajLen(); t++) {
			buff[trajCount].len++; // We have one more transition!
			// Get action
			int a = pi.getAction(s, generator);
			buff[trajCount].actions.push_back(a);
			double actionProbability = pi.getActionProbability(s, a);
			buff[trajCount].actionProbabilities.push_back(actionProbability);

			if (s != 0)
				s = 0;
			else {
				if (p6true(generator)) {
					if (a == 0)
						s = 2;
					else
						s = 1;
				}
				else {
					if (a == 0)
						s = 1;
					else
						s = 2;
				}
			}

			// Update the reward
			double reward = 0;
			if (s == 1)
				reward = 1;
			else if (s == 2)
				reward = -1;
			buff[trajCount].rewards.push_back(reward);
			buff[trajCount].R += reward;

			// Add the state and features for the next element
			buff[trajCount].states.push_back(s);
		}
	}
}

double ModelWin::evaluatePolicy(const Policy & pi, mt19937_64 & generator) const {
	int numSamples = 10000000;
	double result = 0;
	bernoulli_distribution p6true(0.6);
	for (int trajCount = 0; trajCount < numSamples; trajCount++) {
		int s = 0;
		for (int t = 0; t < getMaxTrajLen(); t++) {
			int a = pi.getAction(s, generator);
			if (s != 0)
				s = 0;
			else {
				if (p6true(generator)) {
					if (a == 0)
						s = 2;
					else
						s = 1;
				}
				else {
					if (a == 0)
						s = 1;
					else
						s = 2;
				}
			}

			// Update the reward
			double reward = 0;
			if (s == 1)
				reward = 1;
			else if (s == 2)
				reward = -1;
			result += reward;
		}
	}
	return result / (double)numSamples;
}

Policy ModelWin::getPolicy(int index) const {
	if (index == 1)
		return Policy("p1_ModelWin.txt", getNumActions(), getNumStates());
	if (index == 2)
		return Policy("p2_ModelWin.txt", getNumActions(), getNumStates());
	cerr << "Unknown policy index in ModelWin::getPolicy." << endl;
	exit(1);
}

