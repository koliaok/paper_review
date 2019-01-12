#pragma once
#pragma once

#include <Eigen/Dense>
#include "Policy.h"
#include "Trajectory.h"

using namespace std;
using namespace Eigen;

// Pure virtual class for different environments like the Gridworld

class Environment
{
public:
	virtual int getNumActions() const = 0;
	virtual int getNumStates() const = 0;

	// The workhorse - this function generates trajectories using the provided
	// policy and stores them in buff. It should be threadsafe
	virtual void generateTrajectories(
		vector<Trajectory> & buff, 
		const Policy & pi, 
		int numTraj, 
		mt19937_64 & generator) 
		const = 0;

	// Estimate expected return of this policy.
	virtual double evaluatePolicy(
		const Policy & pi, 
		mt19937_64 & generator) const = 0;

	virtual int getMaxTrajLen() const = 0;
	virtual Policy getPolicy(int index) const = 0;
};