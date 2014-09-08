//Copyright (C) 2007 Lionel Moisan: initial version
//Copyright (C) 2010-2011 Pascal Monasse, Pierre Moulon: genericity, C++ class
//
//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "libOrsa/orsa_model.hpp"
#include "libOrsa/conditioning.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace orsa {

/// Points are normalized according to image dimensions. Subclass must
/// initialize logalpha0_ and take account of normalization in doing that.
/// Matrices \a x1 and \a x2 are 2xn, representing Cartesian coordinates.
OrsaModel::OrsaModel(const Mat &x1, int w1, int h1,
                     const Mat &x2, int w2, int h2)
: x1_(x1.nrow(), x1.ncol()), x2_(x2.nrow(), x2.ncol()),
  N1_(3,3), N2_(3,3), bConvergence(false) {
  assert(2 == x1_.nrow());
  assert(x1_.nrow() == x2_.nrow());
  assert(x1_.ncol() == x2_.ncol());

  NormalizePoints(x1, &x1_, &N1_, w1, h1);
  NormalizePoints(x2, &x2_, &N2_, w2, h2);
  logalpha0_[0] = logalpha0_[1] = 0;
}

/// If multiple solutions are possible, return false.
bool OrsaModel::ComputeModel(const std::vector<size_t> &indices, Model *model,std::vector<float>& weight,int OPmethod) const {
	
	std::vector<Model> models;
	if(OPmethod==-1){//=======method of liu
		Fit(indices, &models, weight,1);
		if(models.size() != 1)
			return false;
		*model = models.front();
		const size_t nData = x1_.ncol();
		std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]

		int iterations=10;
		for (int it=0;it<iterations;it++){
			
			double e_total=0;
			for (size_t i = 0; i <nData; ++i)
			{
				int s;
				double error = Error(*model, i, &s);
				vec_residuals[i] = ErrorIndex(error, i, s);
				e_total+=vec_residuals[i].error*vec_residuals[i].error;
			}
			std::sort(vec_residuals.begin(), vec_residuals.end());
			//==============choosing the best N========//
			int N_best=nData;
			double ratio_best=e_total/(N_best*N_best);

			double e2=0;
			int N=0;
			
			std::cout<<ratio_best<<std::endl;
			for (size_t i = 0; i <nData; ++i)
			{
				e2+=vec_residuals[i].error*vec_residuals[i].error;
				N++;
				if (N>(double)indices.size()/3 && ratio_best>e2/((N-1)*(N-7))){
					ratio_best=e2/((N-1)*(N-7));
					N_best=N;
				}
			}
			//===========optimization============//
			std::cout<<ratio_best<<N_best<<" "<<indices.size()<<std::endl;
			
			std::vector<size_t> vec_inliers(N_best);
			for (size_t i=0; i<N_best; ++i){
				vec_inliers[i] = vec_residuals[i].index;
			}
			Fit(vec_inliers, &models, weight,4);
			if(models.size() == 1) *model = models.front();
		}
		Unnormalize(model);
		return true;
	}							 
	
    Fit(indices, &models, weight,OPmethod);
    if(models.size() != 1)
      return false;
    *model = models.front();
    Unnormalize(model);
    return true;
}



/// logarithm (base 10) of binomial coefficient
static float logcombi(int k, int n)
{
  if (k>=n || k<=0) return(0.0);
  if (n-k<k) k=n-k;
  double r = 0.0;
  for (int i = 1; i <= k; i++)
    r += log10((double)(n-i+1))-log10((double)i);

  return static_cast<float>(r);
}

/// tabulate logcombi(.,n)
static void makelogcombi_n(int n, std::vector<float> & l)
{
  l.resize(n+1);
  for (int k = 0; k <= n; k++)
    l[k] = logcombi(k,n);
}

/// tabulate logcombi(k,.)
static void makelogcombi_k(int k,int nmax, std::vector<float> & l)
{
  l.resize(nmax+1);
  for (int n = 0; n <= nmax; n++)
    l[n] = logcombi(k,n);
}

/// Find best NFA and number of inliers wrt square error threshold in e.
OrsaModel::ErrorIndex OrsaModel::bestNFA(const std::vector<ErrorIndex>& e,
                                         double loge0,
                                         double maxThreshold,
                                         const std::vector<float> &logc_n,
                                         const std::vector<float> &logc_k) const
{
  const int startIndex = SizeSample();
  const double multError = (DistToPoint()? 1.0: 0.5);

  ErrorIndex bestIndex(std::numeric_limits<double>::infinity(),
                       startIndex,
                       0);
  const size_t n = e.size();
  for(size_t k=startIndex+1; k<=n && e[k-1].error<=maxThreshold; ++k) {
    double logalpha = logalpha0_[e[k-1].side] + multError*log10(e[k-1].error);
    ErrorIndex index(loge0+logalpha*(double)(k-startIndex)+logc_n[k]+logc_k[k],
                     k, e[k-1].side);
    if(index.error < bestIndex.error)
      bestIndex = index;
  }
  return bestIndex;
}

/// Denormalize error, recover real error in pixels.
double OrsaModel::denormalizeError(double squareError, int side) const {
  return sqrt(squareError)/(side==0? N1_(0,0): N2_(0,0));
}

/// Get a (sorted) random sample of size X in [0:n-1]
static void random_sample(std::vector<size_t> &k, int X, size_t n)
{
  for(size_t i=0; i < (size_t)X; i++) {
    size_t r = (rand()>>3)%(n-i), j;
    for(j=0; j<i && r>=k[j]; j++)
      r++;
    size_t j0 = j;
    for(j=i; j > j0; j--)
      k[j]=k[j-1];
    k[j0] = r;
  }
}

/// Pick a random sample
/// \param sizeSample The size of the sample.
/// \param vec_index  The possible data indices.
/// \param sample The random sample of sizeSample indices (output).
static void UniformSample(int sizeSample,
                          const std::vector<size_t> &vec_index,
                          std::vector<size_t> *sample) {
  sample->resize(sizeSample);
  random_sample(*sample, sizeSample, vec_index.size());
  for(int i = 0; i < sizeSample; ++i)
    (*sample)[i] = vec_index[ (*sample)[i] ];
}

/// Generic implementation of 'ORSA':
/// A Probabilistic Criterion to Detect Rigid Point Matches
///    Between Two Images and Estimate the Fundamental Matrix.
/// Bibtex :
/// @article{DBLP:journals/ijcv/MoisanS04,
///  author    = {Lionel Moisan and B{\'e}renger Stival},
///  title     = {A Probabilistic Criterion to Detect Rigid Point Matches
///    Between Two Images and Estimate the Fundamental Matrix},
///  journal   = {International Journal of Computer Vision},
///  volume    = {57},
///  number    = {3},
///  year      = {2004},
///  pages     = {201-218},
///  ee        = {http://dx.doi.org/10.1023/B:VISI.0000013094.38752.54},
///  bibsource = {DBLP, http://dblp.uni-trier.de}
///}
/// 
/// ORSA is based on an a contrario criterion of
/// inlier/outlier discrimination, is parameter free and relies on an optimized
/// random sampling procedure. It returns the log of NFA and optionally
/// the best estimated model.
///
/// \param vec_inliers Output vector of inlier indices.
/// \param nIter The number of iterations.
/// \param precision (input/output) threshold for inlier discrimination.
/// \param model The best computed model.
/// \param bVerbose Display optimization statistics.
double OrsaModel::orsa(std::vector<size_t> & vec_inliers,
                       size_t nIter,
                       double *precision,
                       Model *model,
                       bool bVerbose,bool LocOpt) const {
  vec_inliers.clear();

  const int sizeSample = SizeSample();
  const size_t nData = x1_.ncol();
  if(nData <= (size_t)sizeSample)
    return std::numeric_limits<double>::infinity();

  const double maxThreshold = (precision && *precision>0)?
    *precision * *precision *N2_(0,0)*N2_(0,0): // Square max error
    std::numeric_limits<double>::infinity();

  std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]
  std::vector<size_t> vec_sample(sizeSample); // Sample indices

  // Possible sampling indices (could change in the optimization phase)
  std::vector<size_t> vec_index(nData);
  for (size_t i = 0; i < nData; ++i)
    vec_index[i] = i;

  // Precompute log combi
  double loge0 = log10((double)NbModels() * (nData-sizeSample));
  std::vector<float> vec_logc_n, vec_logc_k;
  makelogcombi_n(nData, vec_logc_n);
  makelogcombi_k(sizeSample,nData, vec_logc_k);

  // Output parameters
  double minNFA = std::numeric_limits<double>::infinity();
  double errorMax = 0;
  int side=0;

  // Main estimation loop.
  for (size_t iter=0; iter < nIter; iter++) {
    UniformSample(sizeSample, vec_index, &vec_sample); // Get random sample

    std::vector<Model> vec_models; // Up to max_models solutions
    Fit(vec_sample, &vec_models);

    // Evaluate models
    for (size_t k = 0; k < vec_models.size(); ++k)
    {
      // Residuals computation and ordering
      for (size_t i = 0; i < nData; ++i)
      {
        int s;
        double error = Error(vec_models[k], i, &s);
        vec_residuals[i] = ErrorIndex(error, i, s);
      }
      std::sort(vec_residuals.begin(), vec_residuals.end());

      // Most meaningful discrimination inliers/outliers
      ErrorIndex best = bestNFA(vec_residuals, loge0, maxThreshold,
                                vec_logc_n, vec_logc_k);

      if(best.error < 0 && best.error < minNFA) // A better model was found
      {

        vec_inliers.resize(best.index);
        for (size_t i=0; i<best.index; ++i)
          vec_inliers[i] = vec_residuals[i].index;

		if(LocOpt && vec_inliers.size()>20){
			std::vector<size_t> vec_sample2(20);
			UniformSample(20, vec_inliers, &vec_sample2);

			std::vector<Model> models;
			Fit(vec_sample2, &models ,std::vector<float>(),1);
			
			if(models.size() != 1) 
			{
				if(model) *model = vec_models[k];
			}else{
				if(model) *model = models.front();
				vec_inliers.clear();
				for (size_t i = 0; i < nData; ++i)
				{
					int s;
					double error = Error(*model, i, &s);
					vec_residuals[i] = ErrorIndex(error, i, s);
				}
				std::sort(vec_residuals.begin(), vec_residuals.end());
				best = bestNFA(vec_residuals, loge0, maxThreshold,vec_logc_n, vec_logc_k);
				vec_inliers.resize(best.index);
				for (size_t i=0; i<best.index; ++i)
					vec_inliers[i] = vec_residuals[i].index;

				if (best.error > minNFA && model) {//back check
					vec_inliers.clear();
					for (size_t i = 0; i < nData; ++i)
					{
						int s;
						double error = Error(vec_models[k], i, &s);
						vec_residuals[i] = ErrorIndex(error, i, s);
					}
					std::sort(vec_residuals.begin(), vec_residuals.end());
					best = bestNFA(vec_residuals, loge0, maxThreshold,vec_logc_n, vec_logc_k);
					vec_inliers.resize(best.index);
					for (size_t i=0; i<best.index; ++i)
						vec_inliers[i] = vec_residuals[i].index;
					if(model) *model = vec_models[k];
				}
			}
		}else{
			if(model) *model = vec_models[k];
		}
	  
		minNFA = best.error;
        side = best.side;
        errorMax = vec_residuals[best.index-1].error; // Error threshold
        //if(model) *model = vec_models[k];



        // ORSA optimization: draw samples among best set of inliers so far
		// TODO UNCOMMENT (commented by pierre only for ZHE Liu )
        //vec_index = vec_inliers;
        if(bVerbose)
        {
          std::cout << "  nfa=" << minNFA
                    << " inliers=" << best.index
                    << " precision=" << denormalizeError(errorMax, side)
                    << " im" << side+1
                    << " (iter=" << iter;
          std::cout << ",sample=" << vec_sample.front();
          std::vector<size_t>::const_iterator it=vec_sample.begin();
          for(++it; it != vec_sample.end(); ++it)
            std::cout << ',' << *it;
          std::cout << ")" <<std::endl;
        }
      }
    }
  }

  if(bConvergence)
    refineUntilConvergence(vec_logc_n, vec_logc_k, loge0,
                           maxThreshold, minNFA, model, bVerbose, vec_inliers,
                           errorMax, side);

  if(precision)
    *precision = denormalizeError(errorMax, side);
  if(model && !vec_inliers.empty())
    Unnormalize(model);
  return minNFA;
}
double OrsaModel::ransac(std::vector<size_t> & vec_inliers,
                       size_t nIter,
                       double *precision,
                       Model *model,bool LocOpt) const {
  vec_inliers.clear();

  const int sizeSample = SizeSample();
  const size_t nData = x1_.ncol();
  if(nData <= (size_t)sizeSample)
    return std::numeric_limits<double>::infinity();

  const double maxThreshold = *precision * *precision *N2_(0,0)*N2_(0,0);

  std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]
  std::vector<size_t> vec_sample(sizeSample); // Sample indices

  // Possible sampling indices (could change in the optimization phase)
  std::vector<size_t> vec_index(nData);
  for (size_t i = 0; i < nData; ++i)
    vec_index[i] = i;

  int I_N_best=0; // best number of inliers found so far (store the model that goes with it)
 
  // Main estimation loop.
  for (size_t iter=0; iter < nIter; iter++) {
    UniformSample(sizeSample, vec_index, &vec_sample); // Get random sample

    std::vector<Model> vec_models; // Up to max_models solutions
    Fit(vec_sample, &vec_models);
	
    // Evaluate models
    for (size_t k = 0; k < vec_models.size(); ++k)
    {
		int I_N=0;
		for (size_t i = 0; i < nData; ++i)
		{
			int s;
			double error = Error(vec_models[k], i, &s);
			vec_residuals[i] = ErrorIndex(error, i, s);
			if (error<maxThreshold) I_N++;
		}
	
		if(I_N>I_N_best) // A better model was found
		{
			vec_inliers.clear();
			for (size_t i = 0; i < nData; ++i)
			{
				if (vec_residuals[i].error<maxThreshold) vec_inliers.push_back(i);
			}

			if(LocOpt && vec_inliers.size()>20){
				std::vector<size_t> vec_sample2(20);
				UniformSample(20, vec_inliers, &vec_sample2);

				std::vector<Model> models;
				Fit(vec_sample2, &models ,std::vector<float>(),1);
				if(models.size() != 1){
					if(model) *model = vec_models[k];
				}else{
					if(model) *model = models.front();
					I_N=0;
					vec_inliers.clear();
					for (size_t i = 0; i < nData; ++i)
					{
						int s;
						double error = Error(*model, i, &s);
						if (error<maxThreshold) {
							vec_inliers.push_back(i);
							I_N++;
						}
					}

					if (I_N<=I_N_best){//back check
						I_N=0;
						vec_inliers.clear();
						for (size_t i = 0; i < nData; ++i)
						{
							int s;
							double error = Error(vec_models[k], i, &s);
							if (error<maxThreshold) {
								vec_inliers.push_back(i);
								I_N++;
							}
						}
						if(model) *model = vec_models[k];
					}
				}
			}else{
				if(model) *model = vec_models[k];
			}

			I_N_best=I_N;

		}
	}
  }
  if(model && !vec_inliers.empty())
	  Unnormalize(model);
  return 0;
}

double OrsaModel::mlesac(std::vector<size_t> & vec_inliers,
                       size_t nIter,
                       double *precision,
                       Model *model) const {
  vec_inliers.clear();

  const int sizeSample = SizeSample();
  const size_t nData = x1_.ncol();
  if(nData <= (size_t)sizeSample)
    return std::numeric_limits<double>::infinity();

  const double maxThreshold = (*precision) * (*precision) *N2_(0,0)*N2_(0,0);

  std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]
  std::vector<size_t> vec_sample(sizeSample); // Sample indices

  // Possible sampling indices (could change in the optimization phase)
  std::vector<size_t> vec_index(nData);
  for (size_t i = 0; i < nData; ++i)
    vec_index[i] = i;

 
  double L_best=-100000; // best number of inliers found so far (store the model that goes with it)
  double v=1.0;
  
  // Main estimation loop.
  for (size_t iter=0; iter < nIter; iter++) {
    UniformSample(sizeSample, vec_index, &vec_sample); // Get random sample
    std::vector<Model> vec_models; // Up to max_models solutions
    Fit(vec_sample, &vec_models);
	
    // Evaluate models
    for (size_t k = 0; k < vec_models.size(); ++k)
    {
		double sigma2=0;
		double lambda=0;
		int I_N=0;
		for (size_t i = 0; i < nData; ++i)
		{
			int s;
			double error = Error(vec_models[k], i, &s);
			vec_residuals[i] = ErrorIndex(error, i, s);
			if (error<maxThreshold){ 	
				sigma2+=error;
				I_N++;
			}
		}
		sigma2/=(I_N-7);
		lambda=(double)I_N/nData;
		
		//update lambda
		for(int a=0;a<4;a++){
			double sumLambda=0;
			for (size_t i = 0; i < nData; ++i)
			{
				double pi=lambda*exp(-vec_residuals[i].error/(2*sigma2))/sqrt(2*3.1415926*sigma2);
				double po=(1-lambda)/v;
				sumLambda+=pi/(pi+po);
			}
			lambda=sumLambda/nData;
		}
		//compute L
		double nL=0;
		for (size_t i = 0; i < nData; ++i)
		{
			double pi=lambda*exp(-vec_residuals[i].error/(2*sigma2))/sqrt(2*3.1415926*sigma2);
			double po=(1-lambda)/v;
			nL+= log(pi+po);
		}
		if(nL>L_best) // A better model was found
		{
			L_best=nL;
			vec_inliers.clear();
			for (size_t i = 0; i < nData; ++i)
			{
				if (vec_residuals[i].error<maxThreshold) vec_inliers.push_back(i);
			}
			if(model) *model = vec_models[k];
		}
	}
  }
  if(model && !vec_inliers.empty())
	  Unnormalize(model);
  return 0;
}
//==============================Prosac=======================================================//

// Note on MAX_OUTLIERS_PROPORTION: in this implementation, PROSAC won't stop before having reached the
// corresponding inliers rate on the complete data set.
#define MAX_OUTLIERS_PROPORTION 0.8 // maximum allowed outliers proportion in the input data: used to compute T_N (can be as high as 0.95)
#define P_GOOD_SAMPLE 0.99 // probability that at least one of the random samples picked up by RANSAC is free of outliers

#define TEST_NB_OF_DRAWS 60000 // the max number of draws performed by this test
#define TEST_INLIERS_RATIO 0.5 // The ratio of inliers found by model verification (for the stopping condition)
// beta is the probability that a match is declared inlier by mistake, i.e. the ratio of the "inlier"
// surface by the total surface. The inlier surface is a disc with radius 1.96s for
// homography/displacement computation, or a band with width 1.96*s*2 for epipolar geometry (s is the
// detection noise), and the total surface is the surface of the image.
// YOU MUST ADJUST THIS VALUE, DEPENDING ON YOUR PROBLEM!
#define BETA 0.01

// eta0 is the maximum probability that a solution with more than In_star inliers in Un_star exists and was not found
// after k samples (typically set to 5%, see Sec. 2.2 of [Chum-Matas-05]).
#define ETA0 0.05

/// Computation of the Maximum number of iterations for Ransac
/// with the formula from [HZ] Section: "How many samples" p.119
static inline
int niter_RANSAC(double p, // probability that at least one of the random samples picked up by RANSAC is free of outliers
                 double epsilon, // proportion of outliers
                 int s, // sample size
                 int Nmax) // upper bound on the number of iterations (-1 means INT_MAX)
{
    // compute safely N = ceil(log(1. - p) / log(1. - exp(log(1.-epsilon) * s)))
    double logarg, logval, N;
    if (Nmax == -1) {
        Nmax = INT_MAX;
    }
    assert(Nmax >= 1);
    if (epsilon <= 0.) {
        return 1;
    }
    // logarg = -(1-epsilon)^s
    logarg = -exp(s*log(1.-epsilon)); // C++/boost version: logarg = -std::pow(1.-epsilon, s);
    // logval = log1p(logarg)) = log(1-(1-epsilon)^s)
    logval = log(1.+logarg); // C++/boost version: logval = boost::math::log1p(logarg)
    N = log(1.-p) / logval;
    if (logval  < 0. && N < Nmax) {
        return (int)ceil(N);
    }
    return Nmax;
}
					   
static inline
int Imin(int m, int n, double beta) {
    const double mu = n*beta;
    const double sigma = sqrt(n*beta*(1-beta));
    // Imin(n) (equation (8) can then be obtained with the Chi-squared test with P=2*psi=0.10 (Chi2=2.706)
    return (int)ceil(m + mu + sigma*sqrt(2.706));
}					   
double OrsaModel::prosac(std::vector<size_t> & vec_inliers,	size_t nIter, double *precision, Model *model,bool LocOpt) const {
	//vec_inliers.clear();

	const int sizeSample = SizeSample();
	const size_t nData = x1_.ncol();
	if(nData <= (size_t)sizeSample)
		return std::numeric_limits<double>::infinity();

	const double maxThreshold = *precision * *precision *N2_(0,0)*N2_(0,0);

	std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]
	std::vector<size_t> vec_sample(sizeSample); // Sample indices

	// Precompute log combi
	double loge0 = log10((double)NbModels() * (nData-sizeSample));
	std::vector<float> vec_logc_n, vec_logc_k;
	makelogcombi_n(nData, vec_logc_n);
	makelogcombi_k(sizeSample,nData, vec_logc_k);

	// Output parameters
	double minNFA = std::numeric_limits<double>::infinity();
	double errorMax = 0;
	int side=0;

	const int SAMPLE_SIZE=sizeSample; 

	int CORRESPONDENCES=x1_.ncol();
	int N = CORRESPONDENCES;
	std::vector<int> isInlier(CORRESPONDENCES,0);
	int m = SAMPLE_SIZE;

	int T_N = niter_RANSAC(P_GOOD_SAMPLE, MAX_OUTLIERS_PROPORTION, SAMPLE_SIZE, -1);
	const double beta = BETA;
	int n_star; // termination length (see sec. 2.2 Stopping criterion)
	int I_n_star; // number of inliers found within the first n_star data points
	int I_N_best; // best number of inliers found so far (store the model that goes with it)
	const int I_N_min = (1.-MAX_OUTLIERS_PROPORTION)*N; // the minimum number of total inliers
	int t; // iteration number
	int n; // we draw samples from the set U_n of the top n data points
	double T_n; // average number of samples {M_i}_{i=1}^{T_N} that contain samples from U_n only
	int T_n_prime; // integer version of T_n, see eq. (4)
	int k_n_star; // number of samples to draw to reach the maximality constraint
	int i;
	const double logeta0 = log(ETA0);
	n_star = N;
	I_n_star = 0;
	I_N_best = 0;
	t = 0;
	n = m;
	T_n = T_N;
	for(i=0; i<m; i++) {
		T_n *= (double)(n-i)/(N-i);
	}
	T_n_prime = 1;
	k_n_star = T_N;


	// Main estimation loop.
	 for (size_t iter=0; iter < nIter; iter++){
		int I_N; // total number of inliers for that sample

		// Choice of the hypothesis generation set
		t = t + 1;
		if ((t > T_n_prime) && (n < n_star)) {
			double T_nplus1 = (T_n * (n+1)) / (n+1-m);
			n = n+1;
			T_n_prime = T_n_prime + ceil(T_nplus1 - T_n);
			T_n = T_nplus1;
		}


		if (t > T_n_prime) {
			// during the finishing stage (n== n_star && t > T_n_prime), draw a standard RANSAC sample
			// The sample contains m points selected from U_n at random
			std::vector<size_t> vec_index(n);
			for (size_t i = 0; i < n; ++i)
			 vec_index[i] = i;
			UniformSample(sizeSample, vec_index, &vec_sample); // Get random sample
			//printf("Draw %d points from U_%d... ", m, n);
		}
		else {
			// The sample contains m-1 points selected from U_{n−1} at random and u_n
			//printf("Draw %d points from U_%d and point u_%d... ", m-1, n-1, n);
			std::vector<size_t> vec_index(n-1);
			for (size_t i = 0; i < n-1; ++i)
				vec_index[i] = i;
			UniformSample(sizeSample-1, vec_index, &vec_sample); // Get random sample
			vec_sample.push_back(n-1);
		}

		std::vector<Model> vec_models; // Up to max_models solutions
		Fit(vec_sample, &vec_models);

		// Evaluate models
		for (size_t k = 0; k < vec_models.size(); ++k)
		{
			// Residuals computation and ordering
			I_N=0;
			for (size_t i = 0; i < nData; ++i)
			{
				int s;
				double error = Error(vec_models[k], i, &s);
				if (error<maxThreshold) I_N++;
			}

			if (I_N > I_N_best) {
				int n_best; // best value found so far in terms of inliers ratio
				int I_n_best; // number of inliers for n_best
			
				vec_inliers.clear();
				for (size_t i = 0; i < nData; ++i)
				{
					int s;
					double error = Error(vec_models[k], i, &s);
					if (error<maxThreshold) vec_inliers.push_back(i);
				}

				if(LocOpt && vec_inliers.size()>20){
					std::vector<size_t> vec_sample2(20);
					UniformSample(20, vec_inliers, &vec_sample2);

					std::vector<Model> models;
					Fit(vec_sample2, &models ,std::vector<float>(),1);
					if(models.size() != 1)
						return false;
					*model = models.front();
					
					I_N=0;
					vec_inliers.clear();
					for (size_t i = 0; i < nData; ++i)
					{
						int s;
						double error = Error(*model, i, &s);
						if (error<maxThreshold) {
							vec_inliers.push_back(i);
							I_N++;
						}
					}	
				}else{
					if(model) *model = vec_models[k];
				}

				I_N_best = I_N;
				minNFA = -I_N;
				// Select new termination length n_star if possible, according to Sec. 2.2.
				// Note: the original paper seems to do it each time a new sample is drawn,
				// but this really makes sense only if the new sample is better than the previous ones.
				n_best = N;
				I_n_best = I_N;


				if (1) { // change to if(0) to disable n_star optimization (i.e. draw the same # of samples as RANSAC)
					int n_test; // test value for the termination length
					int I_n_test; // number of inliers for that test value
					double epsilon_n_best = (double)I_n_best/n_best;

					for(n_test = N, I_n_test = I_N; n_test > m; n_test--) { 
						// Loop invariants:
						// - I_n_test is the number of inliers for the n_test first correspondences
						// - n_best is the value between n_test+1 and N that maximizes the ratio I_n_best/n_best
						assert(n_test >= I_n_test);


						if (( I_n_test * n_best > I_n_best * n_test ) &&
							( I_n_test > epsilon_n_best*n_test + sqrt(n_test*epsilon_n_best*(1.-epsilon_n_best)*2.706) )) {
								if (I_n_test < Imin(m,n_test,beta)) {
									// equation 9 not satisfied: no need to test for smaller n_test values anyway
									break; // jump out of the for(n_test) loop
								}
								n_best = n_test;
								I_n_best = I_n_test;
								epsilon_n_best = (double)I_n_best/n_best;
						}

						// prepare for next loop iteration
						I_n_test -= isInlier[n_test-1];
					} // for(n_test ...
				} // n_star optimization

				// is the best one we found even better than n_star?
				if ( I_n_best * n_star > I_n_star * n_best ) {
					double logarg;
					assert(n_best >= I_n_best);
					// update all values
					n_star = n_best;
					I_n_star = I_n_best;
					k_n_star = niter_RANSAC(1.-ETA0, 1.-I_n_star/(double)n_star, m, T_N);
					//printf("new values: n_star=%d, k_n_star=%d, I_n_star=%d, I_min=%d, I_N=%d\n", n_star, k_n_star, I_n_star, Imin(m,n_best,beta),vec_inliers.size());
				}
			} // if (I_N > I_N_best)
		}
  }

  if(model && !vec_inliers.empty())
    Unnormalize(model);
  return minNFA;
}


/// Refine the model on all the inliers with the "a contrario" model
/// The model is refined while the NFA threshold is not stable.
void OrsaModel::refineUntilConvergence(const std::vector<float> & vec_logc_n,
                                       const std::vector<float> & vec_logc_k,
                                       double loge0,
                                       double maxThreshold,
                                       double minNFA,
                                       Model *model,
                                       bool bVerbose,
                                       std::vector<size_t> & vec_inliers,
                                       double & errorMax,
                                       int & side) const
{
  std::cout << "\n\n OrsaModel::refineUntilConvergence(...)\n" << std::endl;
  const size_t nData = x1_.ncol();
  std::vector<ErrorIndex> vec_residuals(nData); // [residual,index]

  bool bContinue = true;
  int iter = 0;
  do{
    std::vector<Model> vec_models; // Up to max_models solutions
    Fit(vec_inliers, &vec_models);

    // Evaluate models
    for (size_t k = 0; k < vec_models.size(); ++k)
    {
      // Residuals computation and ordering
      for (size_t i = 0; i < nData; ++i)
      {
        double error = Error(vec_models[k], i);
        vec_residuals[i] = ErrorIndex(error, i);
      }
      std::sort(vec_residuals.begin(), vec_residuals.end());

      // Most meaningful discrimination inliers/outliers
      ErrorIndex best = bestNFA(vec_residuals, loge0, maxThreshold,
                                vec_logc_n, vec_logc_k);

      if(best.error < 0 && best.error < minNFA) // A better model was found
      {
        minNFA = best.error;
        side = best.side;
        vec_inliers.resize(best.index);
        for (size_t i=0; i<best.index; ++i)
          vec_inliers[i] = vec_residuals[i].index;
        errorMax = vec_residuals[best.index-1].error; // Error threshold
        if(model) *model = vec_models[k];

        if(bVerbose)
        {
          std::cout << "  nfa=" << minNFA
            << " inliers=" << vec_inliers.size()
            << " precision=" << denormalizeError(errorMax, side)
            << " (iter=" << iter << ")\n";
        }
      }
      else
        bContinue = false;
    }
    if (vec_models.empty())
    {
      bContinue = false;
    }
    ++iter;
  }
  while( bContinue );
}

/// Toggle iterative refinement NFA/RMSE.
void OrsaModel::setRefineUntilConvergence(bool value)
{
  bConvergence = value;
}

/// Iterative refinement NFA/RMSE.
bool OrsaModel::getRefineUntilConvergence() const
{
  return bConvergence;
}

} // namespace orsa
