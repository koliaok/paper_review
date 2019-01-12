#pragma once

/*
COMPILING:
	This file relies on the Eigen and Gurobi libraries. To use Eigen, download
	the current version from here: 
	http://eigen.tuxfamily.org/index.php?title=Main_Page
	This code was tested using version 3.3.1.
	Within the downloaded code, only the "Eigen" folder is needed. Make sure
	that this folder is in the include path when compiling.

	Using Gurobi is a bit more complicated. First, you need a license (which is
	free for academics). To learn about Gurobi, go here: http://www.gurobi.com/
	Below are instructions if using Visual Studio 2015. Importantly, this code
	provides quite different results when using some other lower quality open
	source QP solvers - if you use a different solver it is worth trying to
	reproduce the results that should have been provided along with this code.

	Once you have a Gurobi licence and it is installed on your machine, you
	must ensure that gurobi_c++.h is in the include path. For me it is in
	C:\gurobi701\win64\include
	Next you must link to two libraries. In debug mode these are:
	gurobi_c++mdd2015.lib AND C:\gurobi701\win64\lib\gurobi70.lib
	In release mode these are (notice the first is md, not mdd):
	gurobi_c++md2015.lib AND C:\gurobi701\win64\lib\gurobi70.lib
	
	In general, the library used should match the version of Visual Studio and
	the runtime library used (to check which version you are using, see:
	Properties-->C++-->Code Generation-->Runtime Library).

ABOUT:
	This is a reference implementation of the MAGIC estimator, not an optimized
	implementation. That is, is should be relatively simple to follow the code
	and the code should be correct, so that its results can be compared to an
	optimized version.

	See the comments above the MAGIC function for more details about how to 
	use the MAGIC function.

	Using a profiler, it seems like the main slow-down for this code is that
	the importance weights are recomputed every time the DR function is called
	when creating the bootstrap confidence interval. This could be sped up
	by computing these values a single time (the obvious way to do this is to
	include the variable called "rho" in the code below inside the 
	MAGIC_Trajectory object. However, this could make it more difficult for
	a first time user to apply MAGIC (since he or she would have to compute
	these importance weights when passing the data set, D, to MAGIC, or would
	have to know that it is ok to leave these blank for our code to fill it 
	in). So, in summary, I'm leaving this inefficient code in for simplicity.

	This file also includes implementations of other IS estimators: IS, PDIS,
	WIS, CWPDIS, DR, WDR.
*/

#include <vector>
#include <assert.h>
#include <random>
#include <Eigen/Dense>

#include <gurobi_c++.h>

// First some quick prototypes for things that the MAGIC function will use
template <typename RealVector> struct MAGIC_Trajectory;

double cov(const Eigen::VectorXd & a, const Eigen::VectorXd & b);

template <typename RealVector>
double DR(const std::vector<const MAGIC_Trajectory<RealVector>*> & B,
	const bool & useWeightedImportanceSampling, const double & gamma);

Eigen::VectorXd solveQP_onSimplex(const Eigen::MatrixXd & A,
	GRBEnv * env = nullptr);

/*
Run the MAGIC estimator on the provided data set, D, and return the prediction
of the expected discounted return that would result from running the evaluation
policy.

The input "gamma" is the discount parameter.

The input J is the set of return lengths that are considered, $\mathcal J$ in
the MAGIC paper. If the horizon is short and finite, then we recommend using
J = {-2, -1,0, 1,2,...,L-1}, where L is the known finite horizon or longest 
observed trajectory's length. Notice that the values in J can be anything 
greater or equal to -2.

The input delta \in (0,1) is used when constructing the confidence bound on the
importa nce sampling estimator---it is a 1-delta approximate confidence bound. 
We recommend using delta = 0.1 as a first try. Smaller delta means that MAGIC
trusts the model more initially and larger delta means that it trusts the
approximate model less.

The input variable "kappa" is the number of resamplings used by the bootstrap
confidence interval. Ideally it should be closer to 2000, but this is often
extremely slow to run, so we stick with a default value of 200 for now.

The input variable "epsilon" is a scalar that is added to the diagonal of the
matrix in the objective function of the QCQP that we solve. This ensures that
it is well conditioned. We found that in some cases epsilon = 0 works better
than the default value that we usually use, epsilon = 0.00001. The results in
the paper use a mix of these two values for epsilon (since we view this as
an implementation detail, not part of the algorithm).

The original MAGIC paper had an error in the implementation - a factor of n
was missing from the covariance matrix. Setting "includeError" input decides
whether or not this error should be included. Including this error, using
delta \approx 0.1 seems to work well. Not including this error, using
delta \approx 1 seems to work well (this results in 50% confidence intervals
on each side). This version of the code is where this error was spotted. Phil
will reproduce the results of the paper with this corrected version of the
algorithm and will publish an errata as soon as he can.

Recommended types for RealVector are VectorXd, VectorXf, or vector<double>.

Recommended types for IntVector are VectorXi or vector<int>.

Recommended types for Generator are std::mt19937 or std::mt19937_64.

The input mvpie is the model's estimate of the expected return under the
evaluation policy: Model Value PI_E (MVPIE). It is needed if J includes -2. If
J does not include -2, then any value can be passed for mvpie.

If useWeightedImportanceSampling = true, then MAGIC used WDR. If it is false,
then MAGIC uses DR. We recommend using useWeightedImportanceSampling = true.
*/
template <typename RealVector, typename IntVector, typename Generator>
double MAGIC(
	const std::vector<MAGIC_Trajectory<RealVector>> & D,
	const double & gamma,
	const IntVector & J,
	const double & mvpie,
	Generator & generator,
	bool includeError = false,
	const double delta = 0.5,
	const bool & useWeightedImportanceSampling = true,
	const int & kappa = 200,
	const double & epsilon = 0.00001)
{
	// Get the number of trajectories
	int n = (int)D.size();
	if (n < 2)
	{
		cerr << "Error in MAGIC: At least two trajectories are required."
			<< endl;
		assert(false); exit(1);
	}

	// Make sure all of the trajectories look like what we expect
	for (int i = 0; i < n; i++)
	{
		if (D[i].len < 1)
		{
			cerr << "Error in MAGIC: Trajectories must have len >= 1." << endl;
			assert(false); exit(1);
		}
		if (((int)D[i].pib.size() != D[i].len) ||
			((int)D[i].pie.size() != D[i].len) ||
			((int)D[i].q.size() != D[i].len) ||
			((int)D[i].v.size() != D[i].len) ||
			((int)D[i].r.size() != D[i].len))
		{
			cerr << "Error in MAGIC: A vector in a trajectory has unexpected "
				<< "length." << endl;
			assert(false); exit(1);
		}
	}

	// Get the maximum return length that we consider. We don't use Eigen's
	// maxCoeff function so that J can be a vector<int>.
	int L = 0;
	for (int jIndex = 0; jIndex < (int)J.size(); jIndex++)
		L = std::max<int>(L, J[jIndex]+1);
	
	/////
	// Compute all of the importance weights that we will use
	///// 
	// rho(t,i) = $\rho_t^i$ in the paper - the importance weight from time
	// 0 through t in the i'th trajectory.
	Eigen::MatrixXd rho(L, n);
	for (int i = 0; i < n; i++)
		rho(0, i) = D[i].pie[0] / D[i].pib[0];
	for (int t = 1; t < L; t++)
	{
		for (int i = 0; i < n; i++)
		{
			if (t < D[i].len)
				rho(t, i) = rho(t - 1, i) * (D[i].pie[t] / D[i].pib[t]);
			else
			{
				// This episode is in the terminal absorbing state where there
				// is only one action, and so the likelihood ratio used above 
				// will be one.
				rho(t, i) = rho(t - 1, i);
			}
		}
	}

	// w(t,i) = $w_t^i$ in the paper.
	Eigen::MatrixXd w(L, n);
	if (useWeightedImportanceSampling)
	{
		for (int t = 0; t < L; t++)
			w.row(t).array() = rho.row(t).array() / rho.row(t).sum();
	}
	else
		w.array() = rho.array() / (double)n;

	/////
	// Load g with the different length off-policy returns for each trajectory
	///// 
	// Allocate the g matrix and initialize to zero. We will load g(i,j) with
	// the definition of $g_i^{(j)}(D)$ below equation (4) in the paper.
	Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n, (int)J.size());

	// Loop over the rows of g, which index the trajectories
	for (int i = 0; i < n; i++)
	{
		// Loop over the entries in J and compute the J[j] step returns.
		for (int jIndex = 0; jIndex < (int)J.size(); jIndex++)
		{	
			// Get the current return length of interest
			int j = J[jIndex];

			// Use a special case to handle the -2 step return.
			if (j == -2)
			{
				g(i, jIndex) = mvpie / (double) n;
				continue;	// Skip to the next value of jIndex
			}

			// Term (a) in the equation below equation (4). The sum here stops
			// at D[i].len-1, since D[i].r[t] is zero for t >= D[i].len. 
			double curGamma = 1.0;
			for (int t = 0; t <= std::min<int>(j, D[i].len-1); t++)
			{
				g(i,jIndex) += curGamma * w(t, i) * D[i].r[t];
				curGamma *= gamma;
			}

			// Term (b) in the equation below equation (4).
			if (j < D[i].len - 1) // Otherwise v[j+1] = 0 for terminal state.
			{
				// Use special case to set w(-1,i) = 1.
				if (j == -1)
					g(i, jIndex) += curGamma * D[i].v[j + 1] / (double)n;
				else
					g(i, jIndex) += curGamma * w(j, i) * D[i].v[j + 1];
			}

			// Term (c) in the equation below equation (4). The sum here stops
			// at D[i].len-1 since then q[t] and v[t] would be zero since
			// S_t would be terminal.
			curGamma = 1.0;
			for (int t = 0; t <= std::min<int>(j, D[i].len-1); t++)
			{
				// Handle w(-1,i)
				double w2 = (t == 0 ? 1.0/(double)n : w(t - 1, i)); 

				double temp = w(t, i)*D[i].q[t] - w2*D[i].v[t];
				g(i, jIndex) -= curGamma * temp;
				curGamma *= gamma;
			}
		}
	}

	// Grab the mean vector, which gives $g^j(D)$ in the paper
	Eigen::VectorXd gVec = g.colwise().sum();
	
	/////
	// Compute the sample covariance matrix
	/////
	// Be a little inefficient - we could leverage symmetry, but that won't
	// be the bottleneck of this function.
	Eigen::MatrixXd Omega((int)J.size(), (int)J.size());
	for (int i = 0; i < (int)J.size(); i++)
	{
		for (int j = 0; j < (int)J.size(); j++)
			Omega(i, j) = cov(g.col(i), g.col(j));
	}
	// Notice the extra $n$ in the numerator of Equation (5).
	if (!includeError)
		Omega = Omega*(double)n;

	/////
	// Compute the bias vector
	/////
	// First, create a new vector of trajectory pointers that we will use
	// during the bootstrap process
	std::vector<const MAGIC_Trajectory<RealVector>*> B(n);

	// Next, compute the WDR
	for (int i = 0; i < n; i++)
		B[i] = &D[i];
	// gInf = WDR or DR
	double gInf = DR(B, useWeightedImportanceSampling, gamma);

	// Now do the bootstrap resampling to get a confidence interval on the
	// WDR estimator
	Eigen::VectorXd bootstrapEstimates(kappa);
	std::uniform_int_distribution<int> distribution(0, n - 1);
	for (int i = 0; i < kappa; i++)
	{
		for (int j = 0; j < n; j++)
			B[j] = &D[distribution(generator)];
		bootstrapEstimates[i] = DR(B, useWeightedImportanceSampling, gamma);
	}
	// Sort the bootstrap estimates in ascending order
	std::sort(bootstrapEstimates.data(), 
		bootstrapEstimates.data() + bootstrapEstimates.size());
	std::pair<double, double> CI; // Confidence interval around WDR estimator
	CI.first = std::min<double>(gInf, 
						bootstrapEstimates[(int)(kappa*(delta/2.0))]);
	CI.second = std::max<double>(gInf, 
		bootstrapEstimates[(int)(kappa*(1.0-delta/2.0))]);

	// Using the confidence interval, construct the bias vector
	Eigen::VectorXd b = Eigen::VectorXd::Zero((int)J.size());
	for (int jIndex = 0; jIndex < (int)J.size(); jIndex++)
	{
		if (gVec[jIndex] > CI.second)
			b[jIndex] = gVec[jIndex] - CI.second;
		else if (gVec[jIndex] < CI.first)
			b[jIndex] = CI.first - gVec[jIndex];
	}

	/////
	// Compute the weights that minimize our estimate of MSE
	/////
	// Make a matrix Xi that is used by the QCQP that we solve.
	Eigen::MatrixXd Xi = Omega + b*b.transpose() + 
		epsilon*MatrixXd::Identity((int)J.size(), (int)J.size());
	// Solve the LCQP to get the weight vector, x
	Eigen::VectorXd x = solveQP_onSimplex(Xi);

	/////
	// Return the weighted sum of the j-step returns
	/////
	return x.dot(gVec);
}

/////
// Code required by the above MAGIC function
/////

// The MAGIC estimator takes the data as a std::vector of this struct---one per
// trajectory in the data set.
template <typename RealVector>
struct MAGIC_Trajectory
{
	// The number of actions in the trajectory. All of the subsequent vector
	// variables are of length "len". Len must be at least 1.
	int len;

	// Vector containing the probability of the chosen actions under the
	// sampling (behavior) policy.
	RealVector pib;

	// Vector containing the probability of each action under the evaluation
	// policy.
	RealVector pie;

	// Vector containing q(s,a) for all of the observed states and actions,
	// in order. Here q is the model-based (or value-function based) prediction
	// of the state-action value under the *evaluation* policy.
	RealVector q;

	// Vector containing v(s) for all of the observed states and actions,
	// in order. Here v is the model-based (or value-function based) prediction
	// of the state value.
	RealVector v;

	// Vector containing the observed rewards
	RealVector r;
};

// Get the covariance between two vectors (Eigen vectors)
double cov(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
	assert(a.size() == b.size());
	if ((int)a.size() <= 1)
		return 0;
	double muA = a.mean(), muB = b.mean(), temp = 0;
	for (int i = 0; i < (int)a.size(); i++)
		temp += (a[i] - muA)*(b[i] - muB);
	return temp / (double)(a.size() - 1);
}

template <typename RealVector>
double DR(const std::vector<const MAGIC_Trajectory<RealVector>*> & B,
	const bool & useWeightedImportanceSampling, const double & gamma)
{
	// Get the number of trajectories
	int n = (int)B.size();
	
	// Get the maximum trajectory length that we consider. We don't use Eigen's
	// maxCoeff function so that J can be a vector<int>.
	int L = 0;
	for (int i = 0; i < n; i++)
		L = std::max<int>(L, B[i]->len);

	// Compute all of the importance weights that we will use
	Eigen::MatrixXd rho(L, n);
	for (int i = 0; i < n; i++)
		rho(0, i) = B[i]->pie[0] / B[i]->pib[0];
	for (int t = 1; t < L; t++)
	{
		for (int i = 0; i < n; i++)
		{
			if (t < B[i]->len)
				rho(t, i) = rho(t - 1, i) * (B[i]->pie[t] / B[i]->pib[t]);
			else
			{
				// This episode is in the terminal absorbing state where there
				// is only one action, and so the likelihood ratio used above 
				// will be one.
				rho(t, i) = rho(t - 1, i);
			}
		}
	}

	// w(t,i) = $w_t^i$ in the paper.
	Eigen::MatrixXd w(L, n);
	if (useWeightedImportanceSampling)
	{
		for (int t = 0; t < L; t++)
			w.row(t).array() = rho.row(t).array() / rho.row(t).sum();
	}
	else
		w.array() = rho.array() / (double)n;
	
	// Compute Equation (1) in the paper---the DR or WDR estimator
	double result = 0, w2;
	for (int i = 0; i < n; i++)
	{
		double curGamma = 1.0;
		for (int t = 0; t < std::min<int>(L, B[i]->len); t++)
		{
			// The first term
			result += curGamma*w(t, i)*B[i]->r[t];

			// The second term (we re-use the same sums)
			if (t == 0)
				w2 = 1.0 / (double)n;
			else
				w2 = w(t - 1, i);
			result -= curGamma*(w(t, i)*B[i]->q[t] - w2*B[i]->v[t]);
			curGamma *= gamma;
		}
	}

	return result;
}

template <typename RealVector>
double IS(const std::vector<const MAGIC_Trajectory<RealVector>*> & B,
	const bool & useWeightedImportanceSampling, const double & gamma)
{
	// Get the number of trajectories
	int n = (int)B.size();

	// Importance weights and returns
	Eigen::VectorXd IWs = Eigen::VectorXd::Ones(n), 
		R = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < n; i++)
	{
		double curGamma = 1.0;
		for (int t = 0; t < B[i]->len; t++)
		{
			IWs[i] *= B[i]->pie[t] / B[i]->pib[t];
			R[i] += curGamma * B[i]->r[t];
			curGamma *= gamma;
		}
	}
	
	if (useWeightedImportanceSampling)
		return IWs.dot(R) / IWs.sum();
	else
		return IWs.dot(R) / (double)n;
}

template <typename RealVector>
double PDIS(const std::vector<const MAGIC_Trajectory<RealVector>*> & B,
	const bool & useWeightedImportanceSampling, const double & gamma)
{
	// Get the number of trajectories
	int n = (int)B.size();

	// Get the maximum trajectory length in the data set
	int L = 0;
	for (int i = 0; i < n; i++)
		L = std::max<int>(L, B[i]->len);

	// Importance weights and returns
	Eigen::VectorXd IWs = Eigen::VectorXd::Ones(n);
	double curGamma = 1.0, result = 0;
	for (int t = 0; t < L; t++)
	{
		// Do weighted importance sampling for this reward. Start by updating 
		// the importance weight vector
		for (int i = 0; i < n; i++)
		{
			if (t < B[i]->len)
				IWs[i] *= B[i]->pie[t] / B[i]->pib[t];
		}
		double IWSum = IWs.sum();
		for (int i = 0; i < n; i++)
		{			
			if (t < B[i]->len)
			{
				if (useWeightedImportanceSampling)
					result += curGamma * (IWs[i] / IWSum) * B[i]->r[t];
				else
					result += curGamma * (IWs[i] / (double)n) * B[i]->r[t];
			}
		}
		curGamma *= gamma;
	}
	
	return result;
}

/*
Function for solving a QP on the simplex using Gurobi
Minimize x'Ax
Subject to x >= 0
And sum(x) = 1

Assumes that A is square.
*/
Eigen::VectorXd solveQP_onSimplex(const Eigen::MatrixXd & A,
	GRBEnv * env)
{
	int n = (int)A.rows();
	// Track whether or not we created it so that we know if we should del it
	bool createdEnv = false; 
	Eigen::VectorXd result(n);

	try
	{
		// If no GRBEnv provided, create one
		if (env == nullptr)
		{
			createdEnv = true;
			
			// If a Gurobi environment hasn't been provided, create one. 
			// There should usually only be one of these in the program
			env = new GRBEnv();

			// Don't print all the debug info
			env->set(GRB_IntParam_LogToConsole, 0); 
		}
		// Create this optimization problem
		GRBModel model = GRBModel(*env);			

		// Create the variables
		vector<GRBVar> vars(n);
		for (int i = 0; i < n; i++)
			vars[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, 
						((string)("x_") + std::to_string(i)).c_str());

		// Integrate new variables
		model.update();

		// Create the objective expression
		GRBQuadExpr obj = 0.0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				obj += vars[i] * vars[j] * A(i, j);
		}

		// Set it as the objective
		model.setObjective(obj, GRB_MINIMIZE);

		// Add constraints. We already have that all variables are in [0,1]. 
		// Now say that that sum to one
		GRBLinExpr sum = 0;
		for (int i = 0; i < n; i++)
			sum += vars[i];
		model.addConstr(sum, GRB_EQUAL, 1, "simplexConstraint");

		// Optimize model
		model.optimize();

		for (int i = 0; i < n; i++)
			result[i] = vars[i].get(GRB_DoubleAttr_X);
	}
	catch (GRBException e)
	{
		std::cout << "Gurobi Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
#ifdef _MSC_VER
		getchar();
		getchar();
#endif
		exit(1);
	}
	catch (...)
	{
		std::cout << "Gurobi Exception during optimization." << std::endl;
#ifdef _MSC_VER
		getchar();
		getchar();
#endif
		exit(1);
	}

	// Clean up memory
	if (createdEnv)
	{
		// This should happen after the "try" statement, otherwise Gurobi 
		// throws a warning about the environment being deleted too soon.
		delete env;
	}

	return result;
}
