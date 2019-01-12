#include "HybridDomain.h"

#include <iostream>

int HybridDomain::getNumActions() const {
	return 2;
}

int HybridDomain::getNumStates() const {
	return 4;
}

int HybridDomain::getMaxTrajLen() const {
	return 20;
}

void HybridDomain::generateTrajectories(vector<Trajectory> & buff, const Policy & pi, int numTraj, mt19937_64 & generator) const {
	buff.resize(numTraj);
	normal_distribution<double> distribution(0,100);
	bernoulli_distribution p6true(0.6);
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
		// Finished POGrid, now do stochastic domain
		s = 0;
		buff[trajCount].states.push_back(s+1);// state[2]
		for (int t = 2; t < getMaxTrajLen(); t++) {
			buff[trajCount].len++; // We have one more transition!
			// Get action
			int a = pi.getAction(s+1, generator);
			buff[trajCount].actions.push_back(a);
			double actionProbability = pi.getActionProbability(s+1, a);
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
			buff[trajCount].states.push_back(s+1);
		}
	}
}

double HybridDomain::evaluatePolicy(const Policy & pi, mt19937_64 & generator) const {
	double result = 0;
	int numTrials = 10000;
	normal_distribution<double> distribution(0,100);
	bernoulli_distribution p6true(0.6);
	for (int trajCount = 0; trajCount < numTrials; trajCount++) {
		int s = 0;
		for (int t = 0; t < getMaxTrajLen(); t++) {
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

			if (t == 1)
				break;	// Entered a terminal state. Last transition
		}

		// Finished POGrid, now do stochastic domain
		s = 0;
		for (int t = 2; t < getMaxTrajLen(); t++) {
			// Get action
			int a = pi.getAction(s+1, generator);
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
	return result / numTrials;
}

Policy HybridDomain::getPolicy(int index) const {
	if (index == 1)
		return Policy("p1_HybridDomain.txt", getNumActions(), getNumStates());
	if (index == 2)
		return Policy("p2_HybridDomain.txt", getNumActions(), getNumStates());
	cerr << "Unknown policy index in HybridDomain::getPolicy." << endl;
	exit(1);
}
