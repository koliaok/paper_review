#include "Gridworld.h"

#include <iostream>

const int g_GRIDWORLD_SIZE = 4;
const int g_GRIDWORLD_MAX_TRAJLEN = 100;

Gridworld::Gridworld(bool trueHorizon) {
	this->trueHorizon = trueHorizon;
}

int Gridworld::getNumActions() const {
	return 4;
}

int Gridworld::getNumStates() const {
	return g_GRIDWORLD_SIZE*g_GRIDWORLD_SIZE - 1;
}

int Gridworld::getMaxTrajLen() const {
	if (trueHorizon)
		return g_GRIDWORLD_MAX_TRAJLEN;
	else
		return g_GRIDWORLD_MAX_TRAJLEN+1; // Just lie by one, and the model will get partial observability.
}

void Gridworld::generateTrajectories(vector<Trajectory> & buff, const Policy & pi, int numTraj, mt19937_64 & generator) const {
	buff.resize(numTraj);
	for (int trajCount = 0; trajCount < numTraj; trajCount++) {
		buff[trajCount].len = 0;
		buff[trajCount].actionProbabilities.resize(0);
		buff[trajCount].actions.resize(0);
		buff[trajCount].rewards.resize(0);
		buff[trajCount].states.resize(1);
		buff[trajCount].R = 0;
		int x = 0, y = 0;
		buff[trajCount].states[0] = x + y*g_GRIDWORLD_SIZE;
		for (int t = 0; t < g_GRIDWORLD_MAX_TRAJLEN; t++) {
			buff[trajCount].len++; // We have one more transition!
			// Get action
			int action = pi.getAction(buff[trajCount].states[t], generator);
			buff[trajCount].actions.push_back(action);
			double actionProbability = pi.getActionProbability(buff[trajCount].states[t], buff[trajCount].actions[t]);
			buff[trajCount].actionProbabilities.push_back(actionProbability);

			// Get next state and reward
			if ((action == 0) && (x < g_GRIDWORLD_SIZE - 1))
				x++;
			else if ((action == 1) && (x > 0))
				x--;
			else if ((action == 2) && (y < g_GRIDWORLD_SIZE - 1))
				y++;
			else if ((action == 3) && (y > 0))
				y--;

			// Update the reward
			double reward;
			if ((x == 1) && (y == 1))
				reward = -10;
			else if ((x == 1) && (y == 3))
				reward = 1;
			else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))
				reward = 10;
			else
				reward = -1;
			buff[trajCount].rewards.push_back(reward);
			buff[trajCount].R += reward;

			if ((t == g_GRIDWORLD_MAX_TRAJLEN - 1) ||
				((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))) {
				// Entered a terminal state. Last transition
				break;
			}

			// Add the state and features for the next element
			buff[trajCount].states.push_back(x + y*g_GRIDWORLD_SIZE);
		}
	}
}

double Gridworld::evaluatePolicy(const Policy & pi, mt19937_64 & generator) const {
	int numSamples = 10000;

	double result = 0;
	for (int trajCount = 0; trajCount < numSamples; trajCount++) {
		int x = 0, y = 0;
		for (int t = 0; t < g_GRIDWORLD_MAX_TRAJLEN; t++) {
			int action = pi.getAction(x + y*g_GRIDWORLD_SIZE, generator);
			if ((action == 0) && (x < g_GRIDWORLD_SIZE - 1))
				x++;
			else if ((action == 1) && (x > 0))
				x--;
			else if ((action == 2) && (y < g_GRIDWORLD_SIZE - 1))
				y++;
			else if ((action == 3) && (y > 0))
				y--;
			// Update reward
			if ((x == 1) && (y == 1))
				result += -10;
			else if ((x == 1) && (y == 3))
				result += 1;
			else if ((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))
				result += 10;
			else
				result += -1;

			if ((t == g_GRIDWORLD_MAX_TRAJLEN - 1) ||
				((x == g_GRIDWORLD_SIZE - 1) && (y == g_GRIDWORLD_SIZE - 1))) {
				// Entered a terminal state. Last transition
				break;
			}
		}
	}
	return result / (double)numSamples;
}

Policy Gridworld::getPolicy(int index) const {
	if (index == 1)
		return Policy("p1.txt", getNumActions(), getNumStates());
	if (index == 2)
		return Policy("p2.txt", getNumActions(), getNumStates());
	if (index == 3)
		return Policy("p3.txt", getNumActions(), getNumStates());
	if (index == 4)
		return Policy("p4.txt", getNumActions(), getNumStates());
	if (index == 5)
		return Policy("p5.txt", getNumActions(), getNumStates());
	cerr << "Unknown policy index in Gridworld::getPolicy." << endl;
	exit(1);
}
