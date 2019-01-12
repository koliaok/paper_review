#pragma once

#include "Environment.hpp"

class ModelFail : public Environment {
public:
	int getNumActions() const override;
	int getNumStates() const override;

	// The workhorse - this function generates trajectories using the provided
	// policy and stores them in buff. It should be threadsafe
	void generateTrajectories(
		vector<Trajectory> & buff,
		const Policy & pi,
		int numTraj,
		mt19937_64 & generator)
		const override;

	// Estimate expected return of this policy.
	double evaluatePolicy(
		const Policy & pi,
		mt19937_64 & generator) const override;

	int getMaxTrajLen() const override;
	Policy getPolicy(int index) const override;
};