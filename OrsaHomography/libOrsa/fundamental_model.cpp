// Copyright (c) 2007-2011 libmv authors.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "libOrsa/fundamental_model.hpp"
#include "libOrsa/conditioning.hpp"
#include "libOrsa/numeric/poly.h"
#include "third_party/svd/svd.hpp"
using namespace std;
namespace orsa {

/// Constructor, computing logalpha0_
FundamentalModel::FundamentalModel(const Mat &x1, int w1, int h1,
                                   const Mat &x2, int w2, int h2,
                                   bool symError)
: OrsaModel(x1, w1, h1, x2, w2, h2), symError_(symError) {
  double D, A; // Diameter and area of image
  D = sqrt(w1*(double)w1 + h1*(double)h1);
  A = w1*(double)h1;
  logalpha0_[0] = log10(2.0*D/A /N1_(0,0));
  D = sqrt(w2*(double)w2 + h2*(double)h2);
  A = w2*(double)h2;
  logalpha0_[1] = log10(2.0*D/A /N2_(0,0));
}

/// Unnormalize a given model (from normalized to image space).
void FundamentalModel::Unnormalize(Model * model) const  {
  UnnormalizerT::Unnormalize(N1_, N2_, model);
}

/**
 * Build a 9 x n matrix from point matches, where each row is equivalent to the
 * equation x'T*F*x = 0 for a single correspondence pair (x', x). The domain of
 * the matrix is a 9 element vector corresponding to F. In other words, set up
 * the linear system
 *
 *   Af = 0,
 *
 * where f is the F matrix as a 9-vector rather than a 3x3 matrix (row
 * major). If the points are well conditioned and there are 8 or more, then
 * the nullspace should be rank one. If the nullspace is two dimensional,
 * then the rank 2 constraint must be enforced to identify the appropriate F
 * matrix.
 *
 * Note that this does not resize the matrix A; it is expected to have the
 * appropriate size already.
 */
static void EncodeEpipolarEquation(const OrsaModel::Mat &x1,
                                   const OrsaModel::Mat &x2,
                                   const std::vector<size_t> &indices,
                                   OrsaModel::Mat *A) {
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t j = indices[i];
    (*A)(i, 0) = x2(0,j) * x1(0,j);  // 0 represents x coords,
    (*A)(i, 1) = x2(0,j) * x1(1,j);  // 1 represents y coords.
    (*A)(i, 2) = x2(0,j);
    
	(*A)(i, 3) = x2(1,j) * x1(0,j);
    (*A)(i, 4) = x2(1,j) * x1(1,j);
    (*A)(i, 5) = x2(1,j);
    
	(*A)(i, 6) = x1(0,j);
    (*A)(i, 7) = x1(1,j);
    (*A)(i, 8) = 1.0;
  }
}

void FundamentalModel::Fit(const std::vector<size_t> &indices,
	std::vector<Mat> *Fs,std::vector<float> & weight,int OPmethod) const {
		assert(2 == x1_.nrow());
		assert(7 <= x1_.ncol());
		assert(x1_.nrow() == x2_.nrow());
		assert(x1_.ncol() == x2_.ncol());

		// Set up the homogeneous system Af = 0 from the equations x'T*F*x = 0.
		Mat A(indices.size(), 9);
		
		if(indices.size() >= 8) { // 8-point algorithm//optimization
			if(OPmethod==0){//least square
				EncodeEpipolarEquation(x1_, x2_, indices, &A);

				if (weight.size()==indices.size()){
					for (size_t i = 0; i < indices.size(); ++i) {
						for(size_t j=0;j<9;++j){
							A(i, j)*=weight[i];
						}
					}
				}
				libNumerics::vector<double> vecNullspace(9);
				SVDWrapper::Nullspace(A, &vecNullspace);

				libNumerics::matrix<double> F(3,3);
				F.read(vecNullspace);

				// Force the fundamental property if the A matrix has full rank.
				libNumerics::matrix<double> FRank2(3,3);
				SVDWrapper::EnforceRank2_3x3(F, &FRank2);
				Fs->push_back(FRank2);
			}
			if(OPmethod==1){// iterative least square by epipolar distance
				int iteration =10;
				libNumerics::matrix<double> FRank2(3,3);
				for (int i=0;i<iteration;i++){ 
					EncodeEpipolarEquation(x1_, x2_, indices, &A);
					if (i>0){//put distance of epipolar lines 
						for (int i=0;i<indices.size();i++){
							size_t j = indices[i];
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							for(size_t k=0;k<9;++k){
								A(i, k)*=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							}

						}
					}
					libNumerics::matrix<double> F(3,3);
					libNumerics::vector<double> vecNullspace(9);
					SVDWrapper::Nullspace(A, &vecNullspace);

					F.read(vecNullspace);
					
					SVDWrapper::EnforceRank2_3x3(F, &FRank2);

					if (i>=iteration-1) Fs->push_back(FRank2);

				}
			}
			if(OPmethod==2){// M-estimator
				int iteration =10;
				libNumerics::matrix<double> FRank2(3,3);
				double sigma=0;
				for (int i=0;i<iteration;i++){ 
					EncodeEpipolarEquation(x1_, x2_, indices, &A);
					if (i>0){//put distance of epipolar lines 
						for (int i=0;i<indices.size();i++){
							size_t j = indices[i];
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							for(size_t k=0;k<9;++k){
								A(i, k)*=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							}
						}
						libNumerics::matrix<double> Fline(9,1);
						for(size_t k=0;k<9;++k){ Fline(k,0)=FRank2(k/3,k%3); }
						libNumerics::matrix<double> errors=A*Fline;
						
						for (int i=0;i<indices.size();i++){
							sigma+=errors(i)*errors(i);
						}
						sigma=sqrt(sigma/indices.size());
						for (int i=0;i<indices.size();i++){
							if (abs(errors(i))>sigma){
								if(abs(errors(i))>3*sigma){
									for(size_t k=0;k<9;++k){
										A(i, k)=0;
									}
								}else{
									for(size_t k=0;k<9;++k){
										A(i, k)*=sigma/abs(errors(i));
									}
								
								}
							}

						}

					}
					libNumerics::matrix<double> F(3,3);
					libNumerics::vector<double> vecNullspace(9);
					SVDWrapper::Nullspace(A, &vecNullspace);

					F.read(vecNullspace);
					SVDWrapper::EnforceRank2_3x3(F, &FRank2);
					if (i>=iteration-1) Fs->push_back(FRank2);

				}
			}
			if(OPmethod==3){//================= M-sac
				int iteration =10;
				libNumerics::matrix<double> FRank2(3,3);
				vector<size_t>  indices2=indices;

				double sigma=0;
				for (int i=0;i<iteration;i++){
					Mat A2(indices2.size(), 9);
					EncodeEpipolarEquation(x1_, x2_, indices2, &A2);
					if (i>0){//put distance of epipolar lines 
						for (int i=0;i<indices2.size();i++){
							size_t j = indices2[i];
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							for(size_t k=0;k<9;++k){
								A2(i, k)*=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							}
						}
						libNumerics::matrix<double> Fline(9,1);
						for(size_t k=0;k<9;++k){ Fline(k,0)=FRank2(k/3,k%3); }
						libNumerics::matrix<double> errors=A2*Fline;
						
						sigma=0;
						for (int i=0;i<indices2.size();i++){sigma+=errors(i)*errors(i);}
						sigma=sqrt(sigma/(indices2.size()+1));

						for (int i=0;i<indices2.size();i++){
							if (abs(errors(i))>1.96*sigma)
							{	for(size_t k=0;k<9;++k)
								{
									A2(i, k)=0;
								}
							}							
						}

					}
					libNumerics::matrix<double> F(3,3);
					libNumerics::vector<double> vecNullspace(9);
					SVDWrapper::Nullspace(A2, &vecNullspace);

					F.read(vecNullspace);
					SVDWrapper::EnforceRank2_3x3(F, &FRank2);

					if (i<iteration-1 && i>0){
						vector<size_t> all(x1_.ncol(),0); 
						for(size_t i=0;i<x1_.ncol();i++){
							all[i]=i;
						}
						Mat B(x1_.ncol(), 9);
						EncodeEpipolarEquation(x1_, x2_, all, &B);
						for (int j=0;j<x1_.ncol();j++){
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							for(size_t k=0;k<9;++k){
								B(j, k)*=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							}
						}
						libNumerics::matrix<double> Fline(9,1);
						for(size_t k=0;k<9;++k){ Fline(k,0)=FRank2(k/3,k%3); }
						libNumerics::matrix<double> errors=B*Fline;
						indices2.clear();
						for (int i=0;i<x1_.ncol();i++){
							if (abs(errors(i))<=1.96*sigma){
								indices2.push_back(i);
							}							
						}
					}
					if (i>=iteration-1) Fs->push_back(FRank2);
				}
			}
			if(OPmethod==4){// iterative least square by epipolar distance for MLESAC
				int iteration =10;
				double ceilling=3.0*N2_(0,0);
				libNumerics::matrix<double> FRank2(3,3);
				vector<size_t>  indices2=indices;

				for (int i=0;i<iteration;i++){
					Mat A2(indices2.size(), 9);
					EncodeEpipolarEquation(x1_, x2_, indices2, &A2);
					if (i>0){//put distance of epipolar lines 
						for (int i=0;i<indices2.size();i++){
							size_t j = indices2[i];
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							double we=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							for(size_t k=0;k<9;++k){
								A2(i, k)*=we;
							}
						}
					}
					libNumerics::matrix<double> F(3,3);
					libNumerics::vector<double> vecNullspace(9);
					SVDWrapper::Nullspace(A2, &vecNullspace);

					F.read(vecNullspace);
					SVDWrapper::EnforceRank2_3x3(F, &FRank2);

					if (i<iteration-1){
						vector<size_t> all(x1_.ncol(),0); 
						for(size_t i=0;i<x1_.ncol();i++){
							all[i]=i;
						}
						Mat B(x1_.ncol(), 9);
						EncodeEpipolarEquation(x1_, x2_, all, &B);
						for (int j=0;j<x1_.ncol();j++){
							double a1=FRank2(0,0)*x1_(0,j)+FRank2(0,1)*x1_(1,j)+FRank2(0,2);
							double b1=FRank2(1,0)*x1_(0,j)+FRank2(1,1)*x1_(1,j)+FRank2(1,2);
							double a2=FRank2(0,0)*x2_(0,j)+FRank2(1,0)*x2_(1,j)+FRank2(2,0);
							double b2=FRank2(0,1)*x2_(0,j)+FRank2(1,1)*x2_(1,j)+FRank2(2,1);
							double we=sqrt(1.0/(a1*a1+b1*b1)+1.0/(a2*a2+b2*b2));
							for(size_t k=0;k<9;++k){
								B(j, k)*=we;
							}
						}
						libNumerics::matrix<double> Fline(9,1);
						for(size_t k=0;k<9;++k){ Fline(k,0)=FRank2(k/3,k%3); }
						libNumerics::matrix<double> errors=B*Fline;
						indices2.clear();
						for (int i=0;i<x1_.ncol();i++){
							if (abs(errors(i))<=ceilling){
								indices2.push_back(i);
							}							
						}
					}
					if (i>=iteration-1) Fs->push_back(FRank2);
				}
			}
		}
		else
		{
			EncodeEpipolarEquation(x1_, x2_, indices, &A);
			// Find the two F matrices in the nullspace of A.
			Mat F1(3,3), F2(3,3);
			SVDWrapper::Nullspace2_Remap33(A,F1,F2);

			// Then, use the condition det(F) = 0 to determine F. In other words, solve
			// det(F1 + a*F2) = 0 for a.
			double a = F1(0, 0), j = F2(0, 0),
				b = F1(0, 1), k = F2(0, 1),
				c = F1(0, 2), l = F2(0, 2),
				d = F1(1, 0), m = F2(1, 0),
				e = F1(1, 1), n = F2(1, 1),
				f = F1(1, 2), o = F2(1, 2),
				g = F1(2, 0), p = F2(2, 0),
				h = F1(2, 1), q = F2(2, 1),
				i = F1(2, 2), r = F2(2, 2);

			// The coefficients are in ascending powers of alpha, i.e. P[N]*x^N.
			double P[4] = {
				a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
				a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k -
				a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
				a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n -
				a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
				j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p,
			};

			// Solve for the roots of P[3]*x^3 + P[2]*x^2 + P[1]*x + P[0] = 0.
			double roots[3];
			int num_roots = SolveCubicPolynomial(P, roots);

			// Build the fundamental matrix for each solution.
			for (int s = 0; s < num_roots; ++s)
				Fs->push_back(F1 + roots[s] * F2);
		}  
}

/// \param F The fundamental matrix.
/// \param index The point correspondence.
/// \param side In which image is the error measured?
/// \return The square reprojection error.
double FundamentalModel::Error(const Mat &F, size_t index, int* side) const {
  double xa = x1_(0,index), ya = x1_(1,index);
  double xb = x2_(0,index), yb = x2_(1,index);

  double a, b, c, d;
  // Transfer error in image 2
  if(side) *side=1;
  a = F(0,0) * xa + F(0,1) * ya + F(0,2);
  b = F(1,0) * xa + F(1,1) * ya + F(1,2);
  c = F(2,0) * xa + F(2,1) * ya + F(2,2);
  d = a*xb + b*yb + c;
  double err =  (d*d) / (a*a + b*b);
  // Transfer error in image 1
  if(symError_) { // ... but only if requested
    a = F(0,0) * xb + F(1,0) * yb + F(2,0);
    b = F(0,1) * xb + F(1,1) * yb + F(2,1);
    c = F(0,2) * xb + F(1,2) * yb + F(2,2);
    d = a*xa + b*ya + c;
    double err1 =  (d*d) / (a*a + b*b);
    if(err1>err) {
      err = err1;
      if(side) *side=0;
    }
  }
  return err;
}

}  // namespace orsa
