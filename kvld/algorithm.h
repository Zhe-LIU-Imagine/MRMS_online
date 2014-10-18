/** @basic structures implementation
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2007-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <sstream>

#include "demo/libImage/image.hpp"
#include "extras/sift/demo_lib_sift.h"
#include "extras/libNumerics/numerics.h"
#include "extras/libMatch/match.h"

#include "libOrsa/homography_model.hpp"
#include "libOrsa/fundamental_model.hpp"


#include "spline.h"


typedef libNumerics::matrix<double> Matrix;

const double PI = 4.0 * atan(1.0f);
//typedef std::pair<size_t,size_t> Pair;

struct Pair{
	size_t first,second;
	float weight;
	Pair(){}
	Pair(size_t i, size_t j){
		first=i;
		second= j;
	}
	Pair(size_t i, size_t j,float w){
		first=i;
		second= j;
		weight=w;
	}

};
//============================== simplified structure of a point=============================//
//if you set KVLD geometry verification to false, you only need to fill x and y in a point structure
struct PointS{
	float x,y,scale,angle;
	PointS(){}
	PointS(const float& x,const float & y):x(x),y(y){}
	PointS(const float& x,const float& y,const float& angle,const float& scale): x(x),y(y),angle(angle),scale(scale){	}
};

//======= structure containing homography, fundamental matrix and thier criterion ==========//
struct FCrit {
  Matrix F;
  bool homography;
  double thres;
  //========functions==========//
  inline Matrix getMatrix()const{ return F; };
  FCrit(){}
  FCrit(const Matrix& f, double th, bool homo): F(f), thres(th), homography(homo){}

  template <typename T>
  bool operator()(const T& a,const T& b) const {
    return error(a,b)<thres;
  }
  
  template <typename T>
  bool operator()(const T& a, const std::vector<float>& P,float N) const {
    return error(a,P,N)<thres;
  }
  template <typename T>
  double error(const T& a,const T& b) const {
    Matrix m1(3,1),m2(3,1);
    m1(0,0)=a.x;
    m1(1,0)=a.y;
    m1(2,0)=1.0;

    m2(0,0)=b.x;
    m2(1,0)=b.y;
    m2(2,0)=1.0;
    if (!homography){
      Matrix line = F*m1;
      double d1 =std::abs((line.t()*m2)(0,0)/sqrt(line(0,0)*line(0,0)+line(1,0)*line(1,0)));
      line =F.t()*m2;
      double d2 = std::abs((line.t()*m1)(0,0)/sqrt(line(0,0)*line(0,0)+line(1,0)*line(1,0)));
      return std::max(d1,d2);
    }else{
      Matrix mirror = F*m1;
      mirror(0,0)/=mirror(2,0);
      mirror(1,0)/=mirror(2,0);
      return sqrt((mirror(0,0)-m2(0,0))*(mirror(0,0)-m2(0,0))+(mirror(1,0)-m2(1,0))*(mirror(1,0)-m2(1,0)));

    }
  }

  template <typename T>
  double inline error(const T& x1,const T& y1, const T& x2, const T& y2) const {
    Matrix m1(3,1),m2(3,1);
    m1(0)=x1;
    m1(1)=y1;
    m1(2)=1.0;

    m2(0)=x2;
    m2(1)=y2;
    m2(2)=1.0;

    if (!homography){
      Matrix line = F*m1;
      double d1 =std::abs((line.t()*m2)(0,0)/sqrt(line(0,0)*line(0,0)+line(1,0)*line(1,0)));
      line =F.t()*m2;
      double d2 = std::abs((line.t()*m1)(0,0)/sqrt(line(0,0)*line(0,0)+line(1,0)*line(1,0)));
      return std::max(d1,d2);
    }

    Matrix mirror = F*m1;
    mirror(0)/=mirror(2);
    mirror(1)/=mirror(2);
    return sqrt((mirror(0)-m2(0))*(mirror(0)-m2(0))+(mirror(1)-m2(1))*(mirror(1)-m2(1)));

    
  }

  double inline error(const Matrix& M, const std::vector<float>& P,float N) const {
    Matrix He= M.t()*F+F.t()*M;
	double e=0;
	for (int a=0; a<3; a++){
		for (int b=0; b<3; b++) {
			e+=abs(He(a,b));
		}
	}
	e+=N*error(P[0],P[1],P[2],P[3]);
	return e;
  }
  template <typename T>
  void inline perfect_position(const T& x1,const T& y1, T& x2, T& y2) const {
   Matrix m1(3,1),m2(3,1);
    m1(0)=x1;
    m1(1)=y1;
    m1(2)=1.0;

    m2(0)=x2;
    m2(1)=y2;
    m2(2)=1.0;

	if (!homography){
		Matrix line = F*m1;
		double d1 =(line.t()*m2)(0,0)/(line(0,0)*line(0,0)+line(1,0)*line(1,0));
		x2-=line(0)*d1;
		y2-=line(1)*d1;
		return;
	}else{
		Matrix mirror = F*m1;
		x2=mirror(0)/mirror(2);
		y2=mirror(1)/mirror(2);
		return;
	}
  }
};

//=================================== intergral image ======================================//
//It is used to efficiently constructe the pyramide of scale images in KVLD
struct IntegralImages{
	Image<double> map;

	IntegralImages(const Image<float>& I);
  
  inline double operator()(double x1, double y1,double x2,double y2)const{
		return get(x2,y2)-get(x1,y2)-get(x2,y1)+get(x1,y1);
	}
  inline double operator()(double x, double y, double size) const{
    double window=0.5*size;
    return (get(x+window,y+window)-get(x-window,y+window)-get(x+window,y-window)+get(x-window,y-window))/(4*window*window);
  }
private :
  inline double get(double x, double y)const{
		int ix=int(x), iy=int(y);
		double dx=x-ix, dy=y-iy;
		if (dx==0 && dy==0)
			return map(iy,ix);
		if (dx==0)
			return map(iy,ix)*(1-dy)+ map(iy+1,ix)*dy;
		if (dy==0)
			return map(iy,ix)*(1-dx)+ map(iy,ix+1)*dx;

		return map(iy,ix)*(1-dx)*(1-dy)+
			map(iy+1,ix)*dy*(1-dx)+
			map(iy,ix+1)*(1-dy)*dx+
			map(iy+1,ix+1)*dx*dy;
	}
};

///=================================== ORSA functions ======================================//
// display the mean/max error in ORSA regression
static void display_stats(const std::vector<Match>& vec_matchings,
  const std::vector<size_t>& vec_inliers, libNumerics::matrix<double>& F, bool homography);

// Remove multiple "same position" matches
template <typename T>
static void rm_duplicates(T& m);

FCrit Find_Model_comparison(const int w1,const int h1,const int w2,const int h2, const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
	const std::vector<Pair>& matches, bool homography, int RBmethod,int OPmethod);

void  mean_error(const libNumerics::matrix<double>& H,const std::vector<Match>& vec_matchings ,const std::vector<size_t>& vec_inliers, float & e, float& cardinal);

bool ORSA_selective(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2, double& precision,
          libNumerics::matrix<double>& H, std::vector<size_t>& vec_inliers,float& b_e, float& b_N, bool homo, int RBmethod,int OPmethod);

//=============================IO interface, convertion of object types======================//
std::ofstream& writeDetector(std::ofstream& out, const keypoint& vect);

std::ifstream& readDetector(std::ifstream& in,keypoint& point);
//======================================elemetuary operations================================//
template <typename T>
inline T point_distance(const T x1, const T y1, const T x2, const T y2){//distance of points
	float a=x1-x2, b=y1-y2;
	return sqrt(a*a+b*b);
}

template <typename T>
inline float point_distance(const T& P1,const T& P2){//distance of points
		 return point_distance<float>(P1.x, P1.y, P2.x, P2.y);
}

inline bool inside(int w, int h, int x,int y,double radios){
	return (x-radios>=0 && y-radios>=0 && x+radios<w && y+radios<h);
}

//range from 0 to 2*PI
inline bool anglefrom(const float& x, const float& y, float& angle){
	if (x!=0)
		angle=atan(y/x);
	else if (y>0)
		angle=PI/2;
	else if (y<0)
		angle=-PI/2;
	else {
		angle=0;
		return false;
	}
	if (x<0)
		angle+=PI;
	if(angle<0)
		angle+=2*PI;

	assert(angle>=0 && angle<2*PI);
	return true;
}

//range from 0 to PI
inline double angle_difference(const double angle1, const double angle2){
	double angle=angle1-angle2;
	while (angle<0) angle+=2*PI;
	while (angle>=2*PI)	angle-=2*PI;

	assert(angle<=2*PI && angle>=0);
	return std::min(angle,2*PI-angle);
}

inline void max(double* list,double& weight, int size, int& index,int& second_index){
	index=0;
	second_index=-1;
	double best=list[index]-list[index+size/2];

	for (int i=0;i<size;i++){
			double value;
			if(i<size/2) value=list[i]-list[i+size/2];
			else value=list[i]-list[i-size/2];

			if (value>best){
				best=value;
				second_index=index;
				index=i;
			}
	}
	weight=best;
}

template<typename T>
inline void variance(std::vector<T>& list, T& var, T& mean){
	var=0.0;
	mean=std::accumulate(list.begin(), list.end(),0);
	for (std::vector<T>::const_iterator it=list.begin(); it!=list.end();it++){
		var+=(*it-mean)*(*it-mean);
	}
	var = sqrt(var/list.size());
}

template<typename ARRAY>
inline void normalize_weight(ARRAY & weight){
  double total= std::accumulate(weight.begin(), weight.end(), 0.0);
	if (!total==0)
  	for (int i=0; i<weight.size();i++)
	  	weight[i]/=total;
}

template<typename T>
inline float consistent(const T& a1,const T& a2,const T& b1,const T& b2){
	float ax=float(a1.x-a2.x);
	float ay=float(a1.y-a2.y);
	float bx=float(b1.x-b2.x);
	float by=float(b1.y-b2.y);

	float angle1=float(b1.angle-a1.angle);
	float angle2=float(b2.angle-a2.angle);

	float ax1=cos(angle1)*ax-sin(angle1)*ay;
	ax1*=float(b1.scale/a1.scale);
	float ay1=sin(angle1)*ax+cos(angle1)*ay;
	ay1*=float(b1.scale/a1.scale);
	float d1=sqrt(ax1*ax1+ay1*ay1);
	float d1_error=sqrt((ax1-bx)*(ax1-bx)+(ay1-by)*(ay1-by));

	float ax2=float(cos(angle2)*ax-sin(angle2)*ay);
	ax2*=float(b2.scale/a2.scale);

	float ay2=float(sin(angle2)*ax+cos(angle2)*ay);
	ay2*=float(b2.scale/a2.scale);
	float d2=sqrt(ax2*ax2+ay2*ay2);
	float d2_error=sqrt((ax2-bx)*(ax2-bx)+(ay2-by)*(ay2-by));
	float d=std::min(d1_error/std::min(d1,point_distance(b1,b2)),d2_error/std::min(d2,point_distance(b1,b2)));
	return d;
}

float getRange(const Image<float>& I,int a,const float p);

template <typename T>
static void rm_duplicates(T& m) {
  std::sort(m.begin(), m.end());
  typename T::iterator end = std::unique(m.begin(), m.end());
  if(end != m.end()) {
    m.erase(end, m.end());
  }
}

inline float get(const LWImage<float>& map,int spline,double i_x, double i_y){
  float out;
  interpolate_spline(map,spline,i_x, i_y, &out);
  return out;
	}


void get_angle(const IntegralImages& I,const keypoint& key,double& dx,double & dy);

void get_angle(const IntegralImages& I,const float x,const float y, float real_step,double& dx,double & dy);