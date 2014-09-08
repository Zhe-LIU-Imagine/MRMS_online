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
//template<typename T>
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

bool ORSA(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2,
          double& precision,
          libNumerics::matrix<double>& H, std::vector<size_t>& vec_inliers,bool homo, bool addnoise,bool clean,bool usedweight)
{
  const size_t n = vec_matchings.size();
  if(n < 5)
  {
      //std::cerr << "Error: ORSA needs 5 matches or more to proceed" <<std::endl;
      return false;
  }
  size_t n2;
  if(addnoise) n2=n+n/10;
  else n2=n;
  libNumerics::matrix<double> xA(2,n2), xB(2,n2);

  for (size_t i=0; i < n; ++i)
  {
    xA(0,i) = vec_matchings[i].x1;
    xA(1,i) = vec_matchings[i].y1;
    xB(0,i) = vec_matchings[i].x2;
    xB(1,i) = vec_matchings[i].y2;
  }
  
  for(int i=n; i< n2; i++){
	xA(0,i)=float(rand())/RAND_MAX*w1;
	xA(1,i)=float(rand())/RAND_MAX*h1;
	xB(0,i) = float(rand())/RAND_MAX*w2;
    xB(1,i) = float(rand())/RAND_MAX*h2;
  }



  std::auto_ptr< orsa::OrsaModel > modelEstimator;
  if(homo){
	  modelEstimator = std::auto_ptr< orsa::OrsaModel >(
		  new orsa::HomographyModel(xA, w1, h1, xB, w2, h2, true));
  }else{
	  //Fundamental
	  modelEstimator = std::auto_ptr< orsa::FundamentalModel >(new orsa::FundamentalModel(xA, w1, h1, xB, w2, h2, true));
  }

  if(modelEstimator->orsa(vec_inliers, 5000, &precision, &H,false)>0.0)
	  return false;
  //std::cout << "Before refinement: ";
	//display_stats(vec_matchings, vec_inliers, H,homo);
  if (clean){
	  vec_inliers.clear();
	  for(size_t i= 0; i<n;i++)
		vec_inliers.push_back(i);
  }

  //=======================Re-estimate with all inliers=================//
  std::vector<float> weight;
  if(usedweight){
	  for (size_t i= 0; i<vec_inliers.size();i++ ){
		  weight.push_back(vec_matchings[vec_inliers[i]].weight);
	  }
  }

  if( modelEstimator->ComputeModel(vec_inliers,&H, weight) )
  {
	 // std::cout << "After  refinement: ";
    //display_stats(vec_matchings, vec_inliers, H,homo);
  } else{
    //std::cerr << "Warning: error in refinement, result is suspect" <<std::endl;
    return false;
  }
  return true;
}


FCrit Find_Model(const Image<float>& I1,const Image<float>& I2,
	const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
			const std::vector<Pair>& matches,double& precision, bool homography, bool addnoise,bool clean,int method,std::vector<float>& weight){

	return Find_Model(I1.Width(),I1.Height(),I2.Width(),I2.Height(),F1,F2, matches, precision,homography,addnoise,clean,method, weight);

}

FCrit Find_Model(const int w1,const int h1,const int w2,const int h2,
	const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
	const std::vector<Pair>& matches,double& precision, bool homography, bool addnoise,bool clean,int method,std::vector<float>& weight){
		//precision=0 default threshold search, else put given precision ex. 2.0

		//std::cout<<"==========Runing Orsa==========="<<std::endl;

		std::vector<Match> vec_matches;
		bool usedweight;
		if (weight.size()==matches.size()){
			usedweight=true;
			for (size_t i= 0 ; i<matches.size();i++){
				vec_matches.push_back(Match(F1[matches[i].first].x,F1[matches[i].first].y,F2[matches[i].second].x,F2[matches[i].second].y,weight[i]));
			}
		}else{
			usedweight=false;
			for (size_t i= 0 ; i<matches.size();i++){
				vec_matches.push_back(Match(F1[matches[i].first].x,F1[matches[i].first].y,F2[matches[i].second].x,F2[matches[i].second].y));
			}
		}

		rm_duplicates(vec_matches);

		libNumerics::matrix<double> H(3,3);
		std::vector<size_t> vec_inliers;

		bool bRes;
		if(method==0) bRes= ORSA(vec_matches, w1, h1, w2, h2,precision, H, vec_inliers,homography,addnoise,clean,usedweight);
		//if(method==1) bRes= PROSAC(vec_matches, w1, h1, w2, h2,precision, H, vec_inliers,homography,addnoise,clean,usedweight);
		/*std::cout << std::endl << "Orsa estimation :\n"
		<< "precision: " << precision
		<< " validated:" << vec_inliers.size()<<std::endl;*/

		if (homography)
			H/=H(2,2);

		if (!bRes) precision=-1.0;

		return FCrit(H,precision,homography);

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

//===========//
bool ORSA_selective(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2, double& precision,
          libNumerics::matrix<double>& H, std::vector<size_t>& vec_inliers,float& b_e, float& b_N, bool homo,  int RBmethod,int OPmethod)
{
  const size_t n = vec_matchings.size();
  if(n < 5)
  {
      //std::cerr << "Error: ORSA needs 5 matches or more to proceed" <<std::endl;
      return false;
  }
  const size_t steps=12;
  float ratio[steps]={0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};


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

  if(RBmethod==0 && modelEstimator->  orsa(vec_inliers, 5000, &precision, &H,false)>0.0) return false;
  if(RBmethod==1 && modelEstimator->ransac(vec_inliers, 5000, &precision, &H)>0.0) return false;
  if(RBmethod==2 && modelEstimator->ransac(vec_inliers, 5000, &precision, &H,true )>0.0) return false;
  if(RBmethod==3 && modelEstimator->mlesac(vec_inliers, 5000, &precision, &H )>0.0) return false;

  //=======================Re-estimate with all inliers=================//
  std::vector<float> weight;
  for (size_t i= 0; i<vec_inliers.size();i++ ){
	  weight.push_back(vec_matchings[vec_inliers[i]].weight);
  }
  std::sort(weight.begin(),weight.end());

  vector<float> l_N(steps);
  vector<float> l_e(steps);
  vector<libNumerics::matrix<double>> l_c(steps);

  for(int j =0; j<steps;j++){
	  float value= weight[int(ratio[j]*weight.size())-1];
	  std::vector<size_t> sub_inliers;
	  for(size_t i= 0; i<vec_inliers.size();i++){
		  if (vec_matchings[vec_inliers[i]].weight<=value){
			  sub_inliers.push_back(vec_inliers[i]);
		  }
	  }
	  if( modelEstimator->ComputeModel(vec_inliers,&l_c[j],  std::vector<float>(),OPmethod) )
	  {
		  mean_error(l_c[j],vec_matchings, vec_inliers, l_e[j], l_N[j]);
	  } else{
		  l_e[j]=1000;
		  l_N[j]=1;
	  }

  }
  
  int initial=0;
  b_e=l_e[initial];
  b_N=l_N[initial];
  H=l_c[initial];
  while(!_finite(b_e) && initial < steps){
	  b_e=l_e[initial];
	  b_N=l_N[initial];
	  H=l_c[initial];
	  initial++;
  }

  for(int j =0; j<steps;j++){
	  if (pow(b_e,2)/b_N >pow(l_e[j],2)/l_N[j]){
		  b_e=l_e[j];
		  b_N=l_N[j];
		  H=l_c[j];
	  } 
  }

  return true;
}

bool Precision_selective(const std::vector<Match>& vec_matchings, int w1,int h1, int w2,int h2, double& precision,
          libNumerics::matrix<double>& H, std::vector<size_t>& vec_inliers,float& b_e, float& b_N, bool homo,  int RBmethod,int OPmethod)
{
  const size_t n = vec_matchings.size();
  if(n < 5)
  {
      //std::cerr << "Error: ORSA needs 5 matches or more to proceed" <<std::endl;
      return false;
  }
  const size_t steps=12;
  float ratio[steps]={0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};

  vector<float> l_N(steps);
  vector<float> l_e(steps);
  vector<libNumerics::matrix<double>> l_c(steps);
  vector<vector<size_t>> b_inliers(steps);


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

  if(RBmethod==0 && modelEstimator->  orsa(b_inliers[steps-1], 10000, &precision, &l_c[steps-1],false)>0.0) return false;
  if(RBmethod==1 && modelEstimator->ransac(b_inliers[steps-1], 10000, &precision, &l_c[steps-1])>0.0) return false;
  if(RBmethod==2 && modelEstimator->ransac(b_inliers[steps-1], 10000, &precision, &l_c[steps-1],true )>0.0) return false;
  if(RBmethod==3 && modelEstimator->mlesac(b_inliers[steps-1], 10000, &precision, &l_c[steps-1])>0.0) return false;

  //=======================Re-estimate with all inliers=================//


  FCrit crit(l_c[steps-1],precision,homo);
  std::vector<float> weight;
  for (size_t i= 0; i<b_inliers[steps-1].size();i++ ){
	  weight.push_back(crit.error(vec_matchings[b_inliers[steps-1][i]].x1,vec_matchings[b_inliers[steps-1][i]].y1,vec_matchings[b_inliers[steps-1][i]].x2,vec_matchings[b_inliers[steps-1][i]].y2));
  }
  std::sort(weight.begin(),weight.end());

  
  for(int j =0; j<steps-1;j++){
	  //std::cout<<weight.size()<<" "<<int(ratio[j]*weight.size())<<" "<<weight[int(ratio[j]*weight.size())]<<std::endl;
	  double value= weight[int(ratio[j]*weight.size())];
	  
	  if(RBmethod==0 && modelEstimator->  orsa(b_inliers[j], 10000, &value, &l_c[j],false)>0.0){l_e[j]=1000; l_N[j]=1;}
	  if(RBmethod==1 && modelEstimator->ransac(b_inliers[j], 10000, &value, &l_c[j])>0.0) {l_e[j]=1000; l_N[j]=1;}
	  if(RBmethod==2 && modelEstimator->ransac(b_inliers[j], 10000, &value, &l_c[j],true )>0.0) {l_e[j]=1000; l_N[j]=1;}
	  if(RBmethod==3 && modelEstimator->mlesac(b_inliers[j], 10000, &value, &l_c[j])>0.0) {l_e[j]=1000; l_N[j]=1;}

	  if( b_inliers[j].size()>0 && modelEstimator->ComputeModel(b_inliers[j],&l_c[j],  std::vector<float>(),OPmethod) )
	  {
		  mean_error(l_c[j],vec_matchings, b_inliers[j], l_e[j], l_N[j]);
	  }else{
		  l_e[j]=1000;
		  l_N[j]=1;
	  }

  }
  
  int initial=0;
  b_e=l_e[initial];
  b_N=l_N[initial];
  H=l_c[initial];
  vec_inliers=b_inliers[initial];
  while( (b_inliers[initial].size()==0 ||!_finite(b_e)) 
	     && initial < steps){
	  b_e=l_e[initial];
	  b_N=l_N[initial];
	  H=l_c[initial];
	  vec_inliers=b_inliers[initial];
	  initial++;
  }

  for(int j =0; j<steps;j++){
	  if (pow(b_e,2)/b_N >pow(l_e[j],2)/l_N[j]){
		  b_e=l_e[j];
		  b_N=l_N[j];
		  H=l_c[j];
		  vec_inliers=b_inliers[j];
	  } 
  }

  return true;
}


FCrit Find_Model_selective(const int w1,const int h1,const int w2,const int h2,
	const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
	const std::vector<Pair>& matches, float& b_e, float& b_N, bool homography, int RBmethod,int OPmethod){
	
		//std::cout<<"==========Runing Orsa==========="<<std::endl;

		std::vector<Match> vec_matches;

		for (size_t i= 0 ; i<matches.size();i++){
			vec_matches.push_back(Match(F1[matches[i].first].x,F1[matches[i].first].y,F2[matches[i].second].x,F2[matches[i].second].y,matches[i].weight));
		}

		rm_duplicates(vec_matches);

		libNumerics::matrix<double> H(3,3);
		std::vector<size_t> vec_inliers;
		double precision=3.0;
		if (RBmethod==0) precision=0;
		bool bRes= Precision_selective(vec_matches, w1, h1, w2, h2,precision,  H, vec_inliers, b_e, b_N,homography,RBmethod ,OPmethod);

		if (homography)
			H/=H(2,2);

		if (!bRes) precision=-1.0;

		return FCrit(H,precision,homography);

}


bool myfunction (Match i,Match j) { return (i.weight<j.weight);}

//iterative using epipolar distance
FCrit Find_Dynamique2(const int w1,const int h1,const int w2,const int h2,
	const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,
	const std::vector<Pair>& matches, float& b_card, bool homography){
		
		//std::cout<<"==========Runing Orsa==========="<<std::endl;

		std::vector<Match> vec_matches;
		for (size_t i= 0 ; i<matches.size();i++){
			vec_matches.push_back(Match(F1[matches[i].first].x,F1[matches[i].first].y,F2[matches[i].second].x,F2[matches[i].second].y,matches[i].weight));
		}
		rm_duplicates(vec_matches);

		libNumerics::matrix<double> H(3,3);
		std::vector<size_t> vec_inliers;
		double precision=0;
		
		bool bRes= General_opt(vec_matches, w1, h1, w2, h2,precision, H, vec_inliers,homography,0,1);

		if (homography)
			H/=H(2,2);

		if (!bRes) precision=-1.0;

		FCrit first(H,precision,homography);
		
		if (!bRes) {
			precision=-1.0;
			return first;
		}
//=======================dynamic selection================//
		size_t min_N=size_t(0.4*vec_inliers.size());
		if (min_N==0) return first;

		for(vector<Match>::iterator it =vec_matches.begin();it!=vec_matches.end();it++){//weight is no more used.
			it->weight= first.error(it->x1,it->y1,it->x2,it->y2);
		}
		float final_v=0;
		for(int i =0;i<vec_inliers.size();i++){
			final_v+=vec_matches[vec_inliers[i]].weight*vec_matches[vec_inliers[i]].weight;
		}
		final_v/=(vec_inliers.size()-7)*vec_inliers.size();
		b_card=vec_inliers.size();
		
		while (true){//===================================================//
			//reweighting
			for(vector<Match>::iterator it =vec_matches.begin();it!=vec_matches.end();it++){//weight is no more used.
				it->weight= first.error(it->x1,it->y1,it->x2,it->y2);
			}
			std::sort(vec_matches.begin(),vec_matches.end(),myfunction);
			//===selecting
			float e=0;
			for(int i=0; i<min_N; i++){
				e+=vec_matches[i].weight*vec_matches[i].weight;
			}
			float b_v=e/((min_N-7)*min_N);
			float b_N=min_N;

			for(int i=min_N; i<vec_matches.size(); i++){
				e+=vec_matches[i].weight*vec_matches[i].weight;
				if (e/((i-6)*(i+1))<b_v){
					b_v=e/((i-6)*(i+1));
					b_N=(i+1);
				}
			}
			//====estimation===//
			std::vector<Match> b_matches(&vec_matches[0],&vec_matches[b_N]);
			vec_inliers.clear();
			double precision2=0.0;
			bRes= General_opt(b_matches, w1, h1, w2, h2,precision2, H, vec_inliers,homography,0,1,true);
			FCrit second(H,precision2,homography);
			
			//validation
			float final_v2=0;
			for(int i =0;i<vec_inliers.size();i++){
				final_v2+=vec_matches[vec_inliers[i]].weight*vec_matches[vec_inliers[i]].weight;
			}

			final_v2/=(vec_inliers.size()-7)*vec_inliers.size();
			if (final_v>final_v2){
				first=second;
				final_v=final_v2;
				b_card=vec_inliers.size();

			}else {
				return first; 
			}
		}
		return first;
}






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

void rander(const std::vector<float>& in,std::vector<float>& out){
  float mean=0,var=0;
  out.clear();
  for (int i=0; i<in.size();i++)
  {
    mean+= in[i];
    var+=in[i]*in[i];
  }

  mean/=in.size();
  var=sqrt(var/in.size()-mean*mean);

  for (int i=0; i<in.size();i++)
  {
    out.push_back((in[i]-mean)/var);
  }

}



void writeProfil2(const LWImage<float>& I,const keypoint& key, std::vector<float>& feat, int size, float dim){
  feat.clear();
  float real_step=key.scale * dim * 2 /*diameter*/ /(size-1);

  for (int i=-(size-1)/2; i<=(size-1)/2;i++)
  {
    for (int j=-(size-1)/2; j<=(size-1)/2; j++)
    {
      double i_x=key.x+i*real_step;
      double i_y=key.y+j*real_step;

      if (i_x<0.5||i_x>I.w-0.5 || i_y<0.5||i_y>I.h-0.5)
      {
        feat.push_back(0.0f);
      }else
      {
        feat.push_back(get(I,5,i_x,i_y));
      }

    }
  }
  
}

void writeGrid(float x, float y,float scale,const libNumerics::matrix<double>& H1 , std::vector<double>& grid, int size, float dim){
  grid.clear();
  float real_step=scale * dim * 2 /*diameter*/ /(size-1);
   
  for (int i=-(size-1)/2; i<=(size-1)/2;i++)
  {
    for (int j=-(size-1)/2; j<=(size-1)/2; j++)
    {
      double i_x=x+i*real_step;
      double i_y=y+j*real_step;
      libNumerics::matrix<double>  m1(3,1);
      m1(0)=i_x;
      m1(1)=i_y;
      m1(2)=1.0;

      libNumerics::matrix<double>  mirror =H1*m1;
      mirror(0)/=mirror(2);
      mirror(1)/=mirror(2);   

      grid.push_back(mirror(0));
      grid.push_back(mirror(1));   
  
    }
  }
}

void writeProfilH(const LWImage<float>& I,const std::vector<double>& grid, std::vector<float>& feat, int size){
  feat.clear();
  float real_step;
  for (int i=0; i<=(size-1);i++)
  {
    for (int j=0; j<=(size-1); j++)
    {
      double i_x=grid[2*(i*size+j)];
      double i_y=grid[2*(i*size+j)+1];
      
      if (i_x<0.5||i_x>I.w-0.5 || i_y<0.5||i_y>I.h-0.5)
      {
        feat.push_back(0.0f);
       // cout<< i_x<<" "<<i_y<<" "<<0<<endl;
      }else
      {
        feat.push_back(get(I,5,i_x,i_y));
        //cout<< i_x<<" "<<i_y<<" "<<get(I,i_x,i_y)<<endl;
      }

    }
  }
}

void correctionH(const LWImage<float>& I1,const LWImage<float>& I2,
  const libNumerics::matrix<double>& H1,const libNumerics::matrix<double>& H2, keypoint& key1,keypoint& key2,int size){
  
  float dim=3.0f;
  std::vector<float> feat1, feat2,f1,f2;
  
  while (key1.scale<10 && key1.scale<10){
    key1.scale*=2.0;
    key2.scale*=2.0;
  }
  while (key1.scale>100 && key2.scale>100){
    key1.scale/=2.0;
    key2.scale/=2.0;
  }

  while (size<=81)
  {
    writeProfil2( I1, key1, feat1, size,dim);
    rander(feat1,f1);

    libNumerics::matrix<double>  m2(3,1);
    m2(0)=key2.x;
    m2(1)=key2.y;
    m2(2)=1.0;

    libNumerics::matrix<double>  mirror =H2*m2;
    mirror(0)/=mirror(2);
    mirror(1)/=mirror(2);   

    std::vector<double> grid;
    writeGrid(mirror(0), mirror(1),key1.scale, H1 , grid,  size, dim);
    writeProfilH(I2, grid, feat2,size);
    
    //for (int i=0;i<3;i++) {
    //  libNumerics::matrix<double>  test(3,1);
    //  test(0)=grid[2*i];
    //  test(1)=grid[2*i+1];
    //  test(2)=1.0;
    //  libNumerics::matrix<double>  mtest =H2*test;
    //  mtest(0)/=mtest(2);
    //  mtest(1)/=mtest(2);

	  
    //  cout<<mtest(0)<<" "<<mtest(1)<<" "<<get(I2,grid[2*i],grid[2*i+1])<<endl;
    //}

    rander(feat2,f2);
      
    int _i=0,_j=0;
    float Bvalue=0,v=0;

    int rang=2;

    for (int a=rang; a<size-rang;a++)
      for (int b=rang; b<size-rang;b++) 
      {
        Bvalue+=abs(f1[a*size+b]-f2[a*size+b]);
      }


      for (int i=-rang; i<rang;i++)
      {
        for (int j=-rang; j<rang; j++)
        { v=0;
          for (int a=rang; a<size-rang;a++)
          {
            for (int b=rang; b<size-rang;b++)
            {
              v+=abs(f1[a*size+b]-f2[(a+i)*size+b+j]);
            }
          }
          if (v<Bvalue)
            {
              Bvalue=v;
              _i=i;
              _j=j;
            }
        }
      }

      mirror(0)+=_i*dim*2*key1.scale/(size-1);
      mirror(1)+=_j*dim*2*key1.scale/(size-1);
      mirror(2)=1.0;
      libNumerics::matrix<double>  mirror2=H1*mirror;
       
      key2.x=mirror2(0)/mirror2(2);
      key2.y=mirror2(1)/mirror2(2);   
      size=2*size-1;
      dim=max(1.4, dim-0.2);
  }

 // cout<<key1.x<<" "<<key1.y<<" "<<key2.x<<" "<<key2.y<<endl;
}