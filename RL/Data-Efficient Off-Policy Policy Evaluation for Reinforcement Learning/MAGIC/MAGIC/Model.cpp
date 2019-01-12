#include "Model.hpp"
Model::Model() {
}

Model::Model(const vector<Trajectory*> & trajs, int numStates, int numActions, int L, bool JiangStyle) {
	// Get the number of states and actions that have actually been observed
	this->numStates = numStates;
	this->numActions = numActions;
	this->L = L;

	N = (int)trajs.size();

	// Resize everything and set all to zero
	stateActionCounts.resize(numStates);
	stateActionCounts_includingHorizon.resize(numStates);
	stateActionStateCounts.resize(numStates);
	stateActionStateCounts_includingHorizon.resize(numStates);
	P.resize(numStates);
	R.resize(numStates);
	d0.resize(numStates);
	for (int s = 0; s < numStates; s++) {
		stateActionCounts[s].resize(numActions);
		stateActionCounts_includingHorizon[s].resize(numActions);
		stateActionStateCounts[s].resize(numActions);
		stateActionStateCounts_includingHorizon[s].resize(numActions);
		P[s].resize(numActions);
		R[s].resize(numActions);
		d0[s] = 0;
		for (int a = 0; a < numActions; a++) {
			stateActionCounts[s][a] = 0;
			stateActionCounts_includingHorizon[s][a] = 0;
			stateActionStateCounts[s][a].resize(numStates + 1);
			stateActionStateCounts_includingHorizon[s][a].resize(numStates + 1);
			P[s][a].resize(numStates + 1);
			R[s][a].resize(numStates + 1);
			for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
				stateActionStateCounts[s][a][sPrime] = 0;
				stateActionStateCounts_includingHorizon[s][a][sPrime] = 0;
				P[s][a][sPrime] = 0;
				R[s][a][sPrime] = 0;
			}
		}
	}

	// Compute all of the counts, and set R to the sum of rewards from the [s][a][sPrime] transitions
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < trajs[i]->len; j++) {
			int s = trajs[i]->states[j], a = trajs[i]->actions[j], sPrime = (j == trajs[i]->len - 1 ? numStates : trajs[i]->states[j + 1]);
			double r = trajs[i]->rewards[j];

			if (j != L - 1) {
				stateActionCounts[s][a]++;
				stateActionStateCounts[s][a][sPrime]++;
			}
			else {
				// Tested, and we do get here.
				if (sPrime != numStates)
					exit(1);
			}
			stateActionCounts_includingHorizon[s][a]++;
			stateActionStateCounts_includingHorizon[s][a][sPrime]++;

			R[s][a][sPrime] += r;
		}
	}

	// Compute d0
	for (int i = 0; i < N; i++)
		d0[trajs[i]->states[0]] += 1.0 / (double)N;

	// Compute rMin - used by Nan Jiang's model style from Doubly Robust paper
	double rMin = trajs[0]->rewards[0];
	for (int i = 0; i < N; i++) {
		for (int t = 0; t < trajs[i]->len; t++)
			rMin = min(trajs[i]->rewards[t], rMin);
	}

	// Compute P and R
	for (int s = 0; s < numStates; s++) {
		for (int a = 0; a < numActions; a++) {
			for (int sPrime = 0; sPrime < numStates+1; sPrime++) {
				if (stateActionCounts[s][a] == 0) {
					if (JiangStyle)
						P[s][a][sPrime] = (sPrime == s ? 1 : 0); // Self transition
					else
						P[s][a][sPrime] = (sPrime == numStates ? 1 : 0); // Assume termination
				}
				else
					P[s][a][sPrime] = (double)stateActionStateCounts[s][a][sPrime] / (double)stateActionCounts[s][a];

				if (stateActionStateCounts_includingHorizon[s][a][sPrime] == 0) {
					if (JiangStyle)
						R[s][a][sPrime] = rMin;
					else
						R[s][a][sPrime] = 0;							// No data - don't divide by zero
				}
				else
					R[s][a][sPrime] /= (double)stateActionStateCounts_includingHorizon[s][a][sPrime];
			}
		}
	}
}

void Model::loadEvalPolicy(const Policy & pi, const int & L) {
	//cout << "Starting..." << endl;
	// Load actionProbabilities - so we only compute them once
	actionProbabilities.resize(numStates);
	for (int s = 0; s < numStates; s++)
		actionProbabilities[s] = pi.getActionProbabilities(s);

	// Q[t](s,a) = Q(s,a) given that S_t=s, A_t=a, and at t=L the state is absorbing.
	Q.resize(L);
	for (int t = L-1; t >= 0; t--) {
		Q[t] = MatrixXd::Zero(numStates+1, numActions);
		for (int s = 0; s < numStates; s++) {
			for (int a = 0; a < numActions; a++) {
				for (int sPrime = 0; sPrime < numStates + 1; sPrime++) {
					Q[t](s,a) += P[s][a][sPrime] * R[s][a][sPrime];
					if ((sPrime != numStates) && (t != L-1))
						Q[t](s, a) += P[s][a][sPrime] * actionProbabilities[sPrime].dot(Q[t+1].row(sPrime));
				}
			}
		}
	}

	// Load V[t](s) = V(s) given that S_t = s and at t=L the state is absorbing.
	V.resize(L);
	for (int t = 0; t < L; t++) {
		V[t] = VectorXd::Zero(numStates + 1);
		for (int s = 0; s < numStates; s++)
			V[t][s] = actionProbabilities[s].dot(Q[t].row(s));
	}

	evalPolicyValue = 0;
	for (int s = 0; s < numStates; s++)
		evalPolicyValue += d0[s]*V[0][s];

	//cout << "done..." << endl;
}

vector<Trajectory> Model::generateTrajectories(const Policy & pi, int N, mt19937_64 & generator) const {
	vector<VectorXd> _pi(numStates);
	for (int s = 0; s < numStates; s++)
		_pi[s] = pi.getActionProbabilities(s);

	uniform_real_distribution<double> distribution(0,1);
	vector<Trajectory> trajs(N);
	for (int i = 0; i < N; i++) {
		trajs[i].states.resize(0);
		trajs[i].actions.resize(0);
		trajs[i].rewards.resize(0);
		trajs[i].actionProbabilities.resize(0);
		trajs[i].R = 0;

		int state, action, newState;
		double reward;

		// Get initial state
		state = wrand(generator, d0);
		for (int t = 0; true; t++) {
			// Get action
			action = wrand(generator, _pi[state]);

			// Get next state
			newState = wrand(generator, P[state][action]);

			// Get reward
			reward = R[state][action][newState];

			// Add to return
			trajs[i].R += reward;

			// Store in traj
			trajs[i].states.push_back(state);
			trajs[i].actions.push_back(action);
			trajs[i].rewards.push_back(reward);
			trajs[i].actionProbabilities.push_back(_pi[state][action]);

			// Check if episode over
			if ((newState == numStates) || (t == L-1)) {
				trajs[i].len = t+1;
				break;
			}

			// Set state <-- next-state.
			state = newState;
		}
	}
	return trajs;
}