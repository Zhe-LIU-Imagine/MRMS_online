/** @basic structures implementation
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2007-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "algorithm.h"
#include <functional>
#include <numeric>

using namespace std;


IntegralImages::IntegralImages(const Image<float>& I){
		map.Resize(I.Width()+1,I.Height()+1);
		map.fill(0);
		for (int y=0;y<I.Height();y++)
			for (int x=0;x<I.Width();x++){
				map(y+1,x+1)=double(I(y,x))+map(y,x+1)+map(y+1,x)-map(y,x);
			}
	}

float getRange(const Image<float>& I,int a,const float p){
  float range=sqrt(float(3*I.Height()*I.Width())/(p*a*PI));
  std::cout<<"range ="<<range<<std::endl;
  return range;
}

///================================= ORSA functions ======================================//
static void display_stats(const std::vector<Match>& vec_matchings,
  const std::vector<size_t>& vec_inliers,
   libNumerics::matrix<double>& F, bool homography) {
     double l2=0, linf=0;

     if (homography){
       std::vector<size_t>::const_iterator it=vec_inliers.begin();
       for(; it!=vec_inliers.end(); ++it) {
         const Match& m=vec_matchings[*it];
         Matrix x(3,1);
         x(0,0)=m.x1;
         x(0,1)=m.y1;
         x(0,2)=1.0;
         x = F*x;
         x /= x(0,2);
         double e = (m.x2-x(0,0))*(m.x2-x(0,0)) + (m.y2-x(0,1))*(m.y2-x(0,1));
         l2 += e;
         if(linf < e)
           linf = e;
       }
     }else{
       std::vector<size_t>::const_iterator it=vec_inliers.begin();

       for(; it!=vec_inliers.end(); ++it) {
         const Match& m=vec_matchings[*it];
         double a = F(0,0) * m.x1 + F(0,1) * m.y1 + F(0,2);
         double b = F(1,0) * m.x1 + F(1,1) * m.y1 + F(1,2);
         double c = F(2,0) * m.x1 + F(2,1) * m.y1 + F(2,2);
         double d = a*m.x2 + b*m.y2 + c;
         // double e =  (d*d) / (a*a + b*b);
         double e =  (d*d) / (a*a + b*b);
         l2 += e;
         if(linf < e)
           linf = e;
       }
     }
     std::cout << "Average/max error: "
       << sqrt(l2/vec_inliers.size()) << "/"
       << sqrt(linf) <<std::endl;
}


bool General_opt(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2,
	double& precision,
	libNumerics::matrix<double>& H, std::vector<size_t>& vec_inliers,bool homo, int RBmethod,int OPmethod,bool clean=false){
		const size_t n = vec_matchings.size();
		if(n < 5)
		{
			//std::cerr << "Error: ORSA needs 5 matches or more to proceed" <<std::endl;
			return false;
		}
		
		libNumerics::matrix<double> xA(2,n), xB(2,n);

		for (size_t i=0; i < n; ++i)
		{
			xA(0,i) = vec_matchings[i].x1;
			xA(1,i) = vec_matchings[i].y1;
			xB(0,i) = vec_matchings[i].x2;
			xB(1,i) = vec_matchings[i].y2;
		}
		std::auto_ptr< orsa::OrsaModel > modelEstimator;
		if(homo){
			modelEstimator = std::auto_ptr< orsa::OrsaModel >(
				new orsa::HomographyModel(xA, w1, h1, xB, w2, h2, true));
		}else{
			//Fundamental
			modelEstimator = std::auto_ptr< orsa::FundamentalModel >(new orsa::FundamentalModel(xA, w1, h1, xB, w2, h2, true));
		}

		if(RBmethod==0 && modelEstimator->  orsa(vec_inliers, 15000, &precision, &H,false)>0.0) return false;
		if(RBmethod==1 && modelEstimator->ransac(vec_inliers, 15000, &precision, &H      )>0.0) return false;
		if(RBmethod==2 && modelEstimator->ransac(vec_inliers, 15000, &precision, &H,true )>0.0) return false;
		if(RBmethod==3 && modelEstimator->mlesac(vec_inliers, 15000, &precision, &H      )>0.0) return false;
		//=======================Re-estimate with all inliers=================//
		std::vector<float> weight;

		if (clean){
			vec_inliers.clear();
			for(size_t i= 0; i<n;i++)
				vec_inliers.push_back(i);
		}

		if( modelEstimator->ComputeModel(vec_inliers,&H, weight,OPmethod) )
		{
			// std::cout << "After  refinement: ";
			//display_stats(vec_matchings, vec_inliers, H,homo);
		} else{
			//std::cerr << "Warning: error in refinement, result is suspect" <<std::endl;
			return false;
		}
		return true;
}


FCrit Find_Model_comparison(const int w1,const int h1,const int w2,const int h2, const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
	const std::vector<Pair>& matches,  bool homography, int RBmethod,int OPmethod){
		double precision=3.0;
		if(RBmethod==0) precision=0.0; //orsa

		std::vector<Match> vec_matches;
		for (size_t i= 0 ; i<matches.size();i++){
			vec_matches.push_back(Match(F1[matches[i].first].x,F1[matches[i].first].y,F2[matches[i].second].x,F2[matches[i].second].y,matches[i].weight));
		}

		rm_duplicates(vec_matches);

		libNumerics::matrix<double> H(3,3);
		std::vector<size_t> vec_inliers;

		
		bool bRes= General_opt(vec_matches, w1, h1, w2, h2,precision, H, vec_inliers,homography,RBmethod,OPmethod);
	
		if (homography)
			H/=H(2,2);

		if (!bRes) precision=-1.0;

		return FCrit(H,precision,homography);

}

void  mean_error(const libNumerics::matrix<double>& H,const std::vector<Match>& vec_matchings ,const std::vector<size_t>& vec_inliers, float & e, float& cardinal){
	
	e=0;
	cardinal=0;
	FCrit crit(H,0,false);
	
	for(int i=0; i<vec_inliers.size();i++){
			e+=crit.error(vec_matchings[vec_inliers[i]].x1,vec_matchings[vec_inliers[i]].y1,vec_matchings[vec_inliers[i]].x2,vec_matchings[vec_inliers[i]].y2);
			cardinal++;
	}
	e=e/cardinal;
}

bool myfunction (Match i,Match j) { return (i.weight<j.weight);}

//=============================IO interface, convertion of object types======================//

std::ofstream& writeDetector(std::ofstream& out, const keypoint& feature){
  out<<feature.x<<" "<<feature.y<<" "<<feature.scale<<" "<<feature.angle<<std::endl;
  /*for(int i=0;i<128;i++)  
    out<<feature.vec[i]<<" ";
  out<<std::endl;*/
return out;
}

std::ifstream& readDetector(std::ifstream& in,keypoint& point){
  in>>point.x>>point.y>>point.scale>>point.angle;
  //for(int i=0;i<128;i++)  {
  //  in>>point.vec[i];
  //}
return in;
}

void get_angle(const IntegralImages& I,const keypoint& key,double& dx,double & dy){
  int size=5;
  float real_step=key.scale * 1.4 * 2 /*diameter*/ /(size-1);
  dx=0;
  dy=0;
  for (int i=-(size-1)/2; i<=(size-1)/2;i++)
  {
    for (int j=-(size-1)/2; j<=(size-1)/2; j++)
    {

      double i_x=key.x+i*real_step;
      double i_y=key.y+j*real_step;

            if (i_x<real_step||i_x>I.map.Width()-1-real_step
      || i_y<real_step||i_y>I.map.Height()-1-real_step)
      {
      continue;
      }else
      {
      dx+=I( i_x,i_y,real_step)*i;
      dy+=I( i_x,i_y,real_step)*j;
      }

      //if (i_x<2*real_step||i_x>I.map.Width()-1-2*real_step
      //  || i_y<2*real_step||i_y>I.map.Height()-1-2*real_step)
      //{
      //  continue;
      //}else
      //{
      //  double gx= I( i_x+real_step,i_y,real_step)-I( i_x-real_step,i_y,real_step),
      //    gy=I( i_x,i_y+real_step,real_step)-I( i_x,i_y-real_step,real_step),
      //    g=sqrt(gx*gx+gy*gy);

      //  dx+=g*i;
      //  dy+=g*j;
      //}

    }
  }
}

void get_angle(const IntegralImages& I,const float x,const float y, float real_step,float& dx,float & dy){
  dx=0; dy=0;
  for (int i=-1; i<=1;i++)
  {
    for (int j=-1; j<=1; j++)
    {
      double i_x=x+i*real_step;
      double i_y=y+j*real_step;

      if (i_x<real_step||i_x>I.map.Width()-1-real_step
        || i_y<real_step||i_y>I.map.Height()-1-real_step)
      {
        continue;
      }else
      {
        dx+=I( i_x,i_y,real_step)*i;
        dy+=I( i_x,i_y,real_step)*j;
      }
    }
  }
  dx/=5;
  dy/=5;
}
