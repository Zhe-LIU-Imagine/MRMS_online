/** @Converting functions between structures of openCV and KVLD suitable structures
 ** @author Zhe Liu
 **/
/*
Copyright (C) 2007-12 Zhe Liu
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#ifndef MRMS
#define MRMS

#include <algorithm>
#include <memory>
#include <sstream>

#include <cv.hpp>
#include <cxcore.h>
#include <highgui.h>
#include "opencv2/nonfree/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "demo/libImage/image.hpp"
#include "demo/libImage/image_io.hpp"
#include <omp.h>

#include "./kvld/spline.h"
#include "convert.h"
using namespace std;
//using namespace libNumerics;
//======================================basic algorithm===========================================//
template<typename T>
inline T norm2(pair<T,T> x){
  return x.first*x.first+x.second*x.second;
}
template<typename T>
inline T norm2(T x[2]){
  return x[0]*x[0]+x[1]*x[1];
}
template<typename T>
inline T scale(T x[2],T y[2]){
  return x[0]*y[0]+x[1]*y[1];
}
template<typename T>
inline T scale(pair<T,T> x,T y[2]){
  return x.first*y[0]+x.second*y[1];
}
template<typename T>
inline void rotate(T x[2], T angle){
  float t[2]={cos(angle)*x[0]-sin(angle)*x[1],
              sin(angle)*x[0]+cos(angle)*x[1]};
  x[0]=t[0];
  x[1]=t[1];
}

template<typename T>
inline void locate(T x,T y,double M[8],T& destX,T& destY){
  destX=(M[0]*x+M[1]*y+M[2])/(1+M[6]*x+M[7]*y);
  destY=(M[3]*x+M[4]*y+M[5])/(1+M[6]*x+M[7]*y); 
}
template<typename T>
inline void add(double M[8],T dM[8]){
  for (int i= 0; i <8; i++)
    M[i]+=dM[i];
}
template<typename T>
inline void moins(double M[8],T dM[8]){
  for (int i= 0; i <8; i++)
    M[i]-=dM[i];
}

//======================================database setting==================================================//

struct Database{
	int size;
	vector<string> name;
	vector<cv::Mat> I;
	vector<cv::Mat> K;
	vector<cv::Mat> P;
	vector<cv::Mat> R;
	vector<cv::Mat> T;
	
	vector<vector<cv::KeyPoint>> keys;
	vector<cv::Mat> descriptors;
	vector<cv::Mat> Icolor;

	~Database();
	Database(int s);

	Database(string path,int number);
};

//=======================================image scales===============================================//
//We construct the image scales by bluring without subsampling at each level. 

struct ImageChain{
	vector<LWImage<float>> map;
	vector<float> scales;
	int spline;
	int w, h;
	ImageChain(const cv::Mat& I,int spline);
	~ImageChain();
	const LWImage<float>& operator[](size_t ind)const;
	size_t size()const;
};

//======================================Match Refinemetion===========================================//

struct Cell{
  int spline;
  static const int size=15;//the number of rows and cols in a grid
  bool norm;// false do not normalize, true normalize as we do
  float deform;
  float sigma; // =1.1

  float F[size*size];// grid values
  float G[2*size*size];//gradient

  float Va[2];
  float Vb[2];
  float var;
  float rs, rt;

  float real_step;//step in real image
 
  float N;//corrdinate rescaling factor
  float pt[4];// translation in x,y, key scale and angle 
  
  std::vector<float> weight; //weighting parameters 
  std::vector<float> posi;// if not a uniform grid
  //==============constructer without blur=====================// 
  Cell(){}

  Cell(const LWImage<float>& I,const keypoint& key,int sp,float dim,bool norm , float deform,float sigma);

  void creat(const LWImage<float>& I,const keypoint& key,int sp,float dim,bool norm , float deform,float sigma);

  void updateP1(const LWImage<float>& I,float dim);

  inline void ModelP2(const Cell& P1,const LWImage<float>& I2,float scale_ratio, double M[8]){
	  real_step=P1.real_step*scale_ratio;

	  float* indF=F;
	  float* indG=G;
	  //========main interpretation=======//
	  for (int i=-(size-1)/2; i<=(size-1)/2;i++){
		  float signi=(i>0)? 1:-1; 
		  for (int j=-(size-1)/2; j<=(size-1)/2; j++){
			  float signj=(j>0)? 1:-1; 
			  float i_x,i_y;
			  locate(signi*P1.Va[0]*P1.posi[abs(i)]+signj*P1.Vb[0]*P1.posi[abs(j)],
				     signi*P1.Va[1]*P1.posi[abs(i)]+signj*P1.Vb[1]*P1.posi[abs(j)],
					 M,i_x,i_y);
			  i_x=pt[0]+i_x*N;
			  i_y=pt[1]+i_y*N;
			  int a=i+(size-1)/2,b=j+(size-1)/2;
			  int index=a*size+b;
			  //==========0 and 1st order=========//
			  if (i_x<1+real_step||i_x>I2.w-1-real_step || i_y<1+real_step||i_y>I2.h-1-real_step){
				  *(indF++)=0.0f;
				  *(indG++)=0;
				  *(indG++)=0;
			  }else{
				  *(indF++)=get(I2,spline,i_x,i_y);
				  *(indG++)=N*(get(I2,spline,i_x+real_step, i_y          )-get(I2,spline,i_x-real_step, i_y          ))/(2*real_step);
				  *(indG++)=N*(get(I2,spline,i_x,           i_y+real_step)-get(I2,spline,i_x,           i_y-real_step))/(2*real_step);
			  }


		  }
	  }
	  if (norm) normalize();
	  
  }

  inline void findR(const Cell & P1){
		int number=size*size;
		cv::Mat W(number,2,cv::DataType<float>::type);
		float* ptrW = (float*) W.data;
		cv::Mat T(number,1,cv::DataType<float>::type);
		float* ptrT = (float*) T.data;

		
		for (int i=0; i<number;i++){
				ptrW[i*2+0]=F[i];
				ptrW[i*2+1]=1;
				ptrT[i]=P1.F[i];
		}
		
		cv::Mat d=(W.t()*W).inv()*W.t()*T;
		float* ptrD = (float*) d.data;
		rs=ptrD[0];
		rt=ptrD[1];
  }

  inline float diff(const Cell& P2){
	  float res=0;
	  if (norm==true){
		  for (int a=0; a<size;a++){
			  for (int b=0; b<size;b++){ 
				  res+=weight[abs(a-(size-1)/2)]*weight[abs(b-(size-1)/2)]
				  *(F[a*size+b]-P2.F[a*size+b])*(F[a*size+b]-P2.F[a*size+b]);
			  }
		  }
	  }else{
		  assert(var==0 && rs>0);
		  for (int a=0; a<size;a++){
			  for (int b=0; b<size;b++){ 
				  res+=weight[abs(a-(size-1)/2)]*weight[abs(b-(size-1)/2)]
						*(F[a*size+b]-P2.rs*P2.F[a*size+b]-P2.rt)*(F[a*size+b]-P2.rs*P2.F[a*size+b]-P2.rt);
			  }
		  }
	  }
	  return res;
  }
  
private:
	   void inline normalize(){
		   //===mean, var
		   rs=0; rt=0;
		   float mean=0;
		   var=0;
		   int number=size*size;
		   for (int i=0; i<number;i++) { 
			   mean+= F[i];
		   }
		   mean/=number;
		   for (int i=0; i<number;i++){  
			   F[i]=F[i]-mean;
		   }

		   for (int i=0; i<number;i++) {
			   var+=F[i]*F[i];
		   }
		   var= sqrt(var/number);
		   if (var==0) var=1;
		   for (int i=0; i<number;i++){ 
			   F[i]/=var;
		   }
	   }  
	
	   void initial(const LWImage<float>& I,float x,float y, float scale,float angle,int sp,float dim);
};

struct CoupleM{
	Cell P1,P2;
	double M[8]; // P2=H*P1 where M is contains the 8 first elements of the matrix supposing the last is 1
	float size;//could be size, but not.
	float dimension;
	float scale_ratio;

	bool normalize;
	float deform;
	float sigma;
	//================constructers=================//
	CoupleM(bool normalize,float deform, float sigma);

	template<typename T>
	CoupleM(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,const int spline, const float dim, bool normalize, float deform,float sigma);

	template<typename T>
	void initial(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,const int spline, const float dim);
	//=============================================
	template<typename T>
	void update(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,float dim);
	
	template<typename T>
	void  ModelP2(const T& I2);

	//================homography OI======//
	void loadH(const libNumerics::matrix<double>& H);
	libNumerics::matrix<double> toMatrix();
	
	void loadH(double H[8]);
	void toMatrix(double H[8]);
	float diff();
	void estimate_lsm(float dM[8]);
	void estimateAffine(float dM[8]);
	void output(float & x, float & y);
};
/*
LSM methods for comparison
*/
float LSM(const LWImage<float>& I1, const LWImage<float>& I2,const int spline, const keypoint& key1, keypoint& key2);

/*
LSFM method, our extension of LSM
*/
float LSFM(const ImageChain& I1, const ImageChain& I2,const int spline, const keypoint& key1,keypoint& key2, libNumerics::matrix<double>& H=libNumerics::matrix<double>(3,3));

void get_RT(const FCrit& crit, const cv::Mat& K1,const cv::Mat& K2, double norm, cv::Mat& R,cv::Mat& T);

//==================================== Point distribution==========================================//
void Matcher(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::DMatch>& matches,
	const float matching_criterion, const bool bSymmetricMatches, vector<float>& loweScore=vector<float>());


//======================================Match Selection===========================================//
void  Criterion(const FCrit& crit, int w, int h,  const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,const std::vector<Pair>& matches,float& e);

void  mean_error(const FCrit& crit, const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,const std::vector<Pair>& matches, float & e, float& cardinal,int OPmethod=0);

void MatchSelection(const Image<float>& If1, const Image<float>& If2,
	const std::vector<Pair>& matchesSorted, const std::vector<keypoint>& F1, const std::vector<keypoint>& F2,
	float& b_e, float& b_N, FCrit& b_c, bool homography, int RBmethod=0,int OPmethod=0,  float rate=2,bool second_check=true);
#endif