#include <iostream>
#include <fstream>

#include "MAGIC.hpp"

#include "Gridworld.h"
#include "ModelFail.hpp"
#include "ModelWin.hpp"
#include "HybridDomain.h"

#include "Model.hpp"

using namespace std;
using namespace Eigen;

int makePlotData(
	const Environment * env, // Pointer to environment obj to use
	const int pibIndex, // Policy index of behavior policy (see env->getPolicy)
	const int pieIndex, // Policy index of evaluation policy
	const char * filename, // File name to print data to
	const int N, // How many different number of trajectories to test?
	const bool MAGIC_includeError,
	const double MAGIC_delta)
{
	// Create the random number generator
	mt19937_64 generator;
		
	// What should the model do if a state-action pair was never observed?
	// Thomas and Brunskill's MAGIC paper handles this one way, and Nan Jian
	// and Lihong Li's Doubly Robust paper handles it a different way. This
	// selects between these two methods.
	bool JiangStyleModel = false;

	// Load numberOfTrajectories with the horizontal axis points that will be
	// tested.
	VectorXi numberOfTrajectories(N);
	numberOfTrajectories[0] = 2;
	for (int i = 1; i < N; i++)
		numberOfTrajectories[i] = 3 + (int)(pow(2, i));

	// Get the behavior and evaluation policies
	Policy pib = env->getPolicy(pibIndex),
		pie = env->getPolicy(pieIndex);

	// Compute the target value using Monte Carlo simulations
	double target = env->evaluatePolicy(pie, generator);

	// Select the value for gamma. The approximate model code here only
	// supports gamma = 1.0 for now, so don't change this. Also, for a finite
	// horizon problem, would you ever really not want gamma = 1?
	double gamma = 1.0;

	// MAGIC uses J to hold the different length returns that we consider. Here
	// we will use all of them, from -2 to L-1, where L is the maximum
	// trajectory length.
	VectorXi J(env->getMaxTrajLen() + 2);
	for (int i = 0; i < env->getMaxTrajLen() + 2; i++)
		J[i] = i - 2;

	// We will store results for 8 different methods, and for each point in
	// the resulting plot (method and number of trajectories) we run M trials.
	int numMethods = 8, M = 128;

	// Create a matrix to store the mean squared error (MSE) of each method
	// with different numbers of trajectories
	MatrixXd MSEs(N, numMethods);

	// Run the actual trials, looping over the number of trajectories
	for (int numTrajIndex = 0; numTrajIndex < N; numTrajIndex++)
	{
		// Compute MSE using n trajectories of historical data
		int n = numberOfTrajectories[numTrajIndex];

		// Create vectors to hold the results from M trials of each of the methods
		VectorXd outAM(M), outIS(M), outPDIS(M), outWIS(M), outCWPDIS(M), 
			outDR(M), outWDR(M), outMAGIC(M);

		// Parallelize the execution of all of the trials
		#pragma omp parallel for num_threads(6) // num_threads(32)
		for (int trial = 0; trial < M; trial++)
		{
			// Create the data (use the same sample of data for each method -
			// a form of common random numbers)
			vector<Trajectory> D;
			env->generateTrajectories(D, pib, n, generator);

			// Store the trajectories as trajectory pointers for faster
			// resamplinenv->
			vector<Trajectory*> D_pointers(n);
			for (int i = 0; i < n; i++)
				D_pointers[i] = &D[i];

			// Create the model from the data
			Model m(D_pointers, env->getNumStates(), env->getNumActions(),
				env->getMaxTrajLen(), JiangStyleModel);

			// Load the value estimates within the model based on the 
			// evaluation policy
			m.loadEvalPolicy(pie, env->getMaxTrajLen());

			// The trajectories are in the form used by the original MAGIC
			// code. We need to put them into the MAGIC_Trajectory form
			// used by the new MAGIC code. Copy them over.
			vector<MAGIC_Trajectory<vector<double>>> D2(n);
			for (int i = 0; i < n; i++)
			{
				D2[i].len = D[i].len;
				D2[i].pib = D[i].actionProbabilities;
				D2[i].r = D[i].rewards;
				D2[i].pie.resize(D[i].len);
				D2[i].q.resize(D[i].len);
				D2[i].v.resize(D[i].len);
				for (int t = 0; t < D[i].len; t++)
				{
					int state = D[i].states[t], action = D[i].actions[t];
					D2[i].pie[t] = pie.getActionProbability(state, action);
					D2[i].q[t] = m.Q[t](state, action);
					D2[i].v[t] = m.V[t](state);
				}
			}

			// Load B with pointers to the MAGIC_Trajectories in D2.
			vector<const MAGIC_Trajectory<vector<double>>*> B(n);
			for (int i = 0; i < n; i++)
				B[i] = &D2[i];

			// Compute the predictions of the different methods. The true and
			// false terms below are for the bool weighted, which sets whether
			// the method uses ordinary importance sampling or weighted
			// importance samplinenv->
			outAM[trial] = m.evalPolicyValue;
			outIS[trial] = IS(B, false, gamma);
			outPDIS[trial] = PDIS(B, false, gamma);
			outWIS[trial] = IS(B, true, gamma);
			outCWPDIS[trial] = PDIS(B, true, gamma);
			outDR[trial] = DR(B, false, gamma);
			outWDR[trial] = DR(B, true, gamma);
			outMAGIC[trial] = MAGIC(D2, gamma, J,m.evalPolicyValue, generator,
				MAGIC_includeError, MAGIC_delta);
		}

		// Convert the outputs to MSEs
		outAM	  = outAM.array() - target;
		outIS	  = outIS.array() - target;
		outPDIS	  = outPDIS.array() - target;
		outWIS	  = outWIS.array() - target;
		outCWPDIS = outCWPDIS.array() - target;
		outDR	  = outDR.array() - target;
		outWDR	  = outWDR.array() - target;
		outMAGIC  = outMAGIC.array() - target;
		MSEs(numTrajIndex, 0) = outAM.dot(outAM)/(double)N;
		MSEs(numTrajIndex, 1) = outIS.dot(outIS) / (double)N;
		MSEs(numTrajIndex, 2) = outPDIS.dot(outPDIS) / (double)N;
		MSEs(numTrajIndex, 3) = outWIS.dot(outWIS) / (double)N;
		MSEs(numTrajIndex, 4) = outCWPDIS.dot(outCWPDIS) / (double)N;
		MSEs(numTrajIndex, 5) = outDR.dot(outDR) / (double)N;
		MSEs(numTrajIndex, 6) = outWDR.dot(outWDR) / (double)N;
		MSEs(numTrajIndex, 7) = outMAGIC.dot(outMAGIC) / (double)N;
	}

	// Print MSEs to output file
	ofstream out(filename);
	out << "numTraj\tAM\tIS\tPDIS\tWIS\tCWPDIS\tDR\tWDR\tMAGIC\n";
	for (int i = 0; i < N; i++) {
		out << numberOfTrajectories[i] << '\t';
		for (int j = 0; j < 8; j++) {
			out << MSEs(i, j);
			if (j != 7)
				out << '\t';
		}
		out << endl;
	}
	out.close();
}

int main(int argc, char * argv[])
{
	// Run with the error from the original code
	cout << "Running trials WITH the error in MAGIC..." << endl;
	Environment * env = nullptr;

	bool trueHorizon = false;
	bool MAGIC_includeError = true;
	double MAGIC_delta = 0.05;

	env = new Gridworld(trueHorizon);
	cout << "Generating data for Gridworld plot..." << endl;
	makePlotData(env, 4, 5, "out_Gridworld_p4p5_err_MSE.txt", 12, MAGIC_includeError, MAGIC_delta);
	delete env;

	/*
	env = new ModelFail();
	cout << "Generating data for ModelFail plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelFail_err_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new ModelWin();
	cout << "Generating data for ModelWin plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelWin_err_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new HybridDomain();
	cout << "Generating data for HybridDomain plot..." << endl;
	makePlotData(env, 1, 2, "out_HybridDomain_err_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	// Run without the error from the original code
	cout << "Running trials WITHOUT the error in MAGIC..." << endl;
	MAGIC_includeError = false;
	MAGIC_delta = 0.5;

	env = new Gridworld(trueHorizon);
	cout << "Generating data for Gridworld plot..." << endl;
	makePlotData(env, 4, 5, "out_Gridworld_p4p5_MSE.txt", 12, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new ModelFail();
	cout << "Generating data for ModelFail plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelFail_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new ModelWin();
	cout << "Generating data for ModelWin plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelWin_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new HybridDomain();
	cout << "Generating data for HybridDomain plot..." << endl;
	makePlotData(env, 1, 2, "out_HybridDomain_MSE.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	// Run without the error from the original code, and with different delta from run above
	cout << "Running trials WITHOUT the error in MAGIC..." << endl;
	MAGIC_includeError = false;
	MAGIC_delta = 0.25;

	env = new Gridworld(trueHorizon);
	cout << "Generating data for Gridworld plot..." << endl;
	makePlotData(env, 4, 5, "out_Gridworld_p4p5_MSE_025.txt", 12, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new ModelFail();
	cout << "Generating data for ModelFail plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelFail_MSE_025.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new ModelWin();
	cout << "Generating data for ModelWin plot..." << endl;
	makePlotData(env, 1, 2, "out_ModelWin_MSE_025.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;

	env = new HybridDomain();
	cout << "Generating data for HybridDomain plot..." << endl;
	makePlotData(env, 1, 2, "out_HybridDomain_MSE_025.txt", 16, MAGIC_includeError, MAGIC_delta);
	delete env;
	*/

	// Tell the user that we're done
	cout << "Done." << endl;

#ifdef _MSC_VER
	// If you are on Windows using visual studio, the terminal will auto-close.
	// Use getchar to let the user choose to close the terminal.
	getchar();
#endif
}