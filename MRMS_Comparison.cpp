/** @Image calibration application
 ** @Estimate fundamental or homography matrix
 ** @author Zhe Liu
 **/
/*
Copyright (C) 2011-2013 Zhe Liu.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).

This is the general PS + PR improvements comparison with 5 Ransac methods 


*/

#include "kvld/kvld.h"
#include "convert.h"
#include "MRMS.h"
#include <numeric>
static const int spline=5;
const float sift_matching_criterion=0.8;


//===============================================================================================================================//
bool myrank (Pair i,Pair j) {return (i.weight<j.weight);}

/*
The function evaluate the performance of 
1. RANSAC methods
2. RANSAC with Match Selection
3. RANSAC with Match Refinement
4. RANSAC with Match Refinement and Match Selection

Five RANSAC methods are available for test.

Input: 

data: structure containing the images, features and ground truth information
ind1: index of the first image in the dataset
ind2: index of the second image in the dataset
crit: vectors of F matrices to be estimated by each method and each iteration.
      crit[a][b] is the result for method ID a, iteration ID b.
homography: if the model to estimate is an homography
iterations: number of iteration for each method

Output:

numbers of kept matches for each method.

*/
vector<int> Comparing(const Database& data, int ind1, int ind2, 
	vector<vector<FCrit>>& crit, bool homography, int iterations, ofstream& msg){
	crit.clear();

	const int items=1;// only ORSA+ IRLS
	int RBmethods[items]={0};
	int OPmethods[items]={1};

	const std::vector<cv::KeyPoint>& feat1=data.keys[ind1];
	const std::vector<cv::KeyPoint>& feat2=data.keys[ind2];
	const cv::Mat& descriptors1=data.descriptors[ind1];
	const cv::Mat& descriptors2=data.descriptors[ind2];	
	//=============== compute matches using brute force matching ====================//
	std::vector<cv::DMatch> matches;
	std::vector<float> lowe_thres;
	bool bSymmetricMatches = true;
	Matcher(descriptors1, descriptors2, matches, sift_matching_criterion, bSymmetricMatches);

	std::vector<keypoint> F1, F2;
	Convert_detectors(feat1,F1);
	Convert_detectors(feat2,F2);

	//=============== convert openCV sturctures to KVLD recognized elements
	Image<float> If1, If2;
	Convert_image(data.I[ind1], If1);
	Convert_image(data.I[ind2], If2);
	std::vector<Pair> matchesPair;
	Convert_matches(matches,matchesPair);

	LWImage<float> I1=Convert_image(data.I[ind1]);
	prepare_spline(I1,spline);
	LWImage<float> I2=Convert_image(data.I[ind2]);
	prepare_spline(I2,spline);

	ImageChain chain1(data.I[ind1],spline);
	ImageChain chain2(data.I[ind2],spline);

//================================================================//




	//===============================  KVLD method ==================================//
	std::cout<<"VLD starts with "<<matches.size()<<" matches"<<std::endl;

	std::vector<Pair> matchesFiltered;
	std::vector<double> vec_score;

	libNumerics::matrix<float> E = libNumerics::matrix<float>::ones(matches.size(),matches.size())*(-1);
	std::vector<bool> valide(matches.size(), true);
	size_t it_num=0;
	KvldParameters kvldparameters;//initial parameters of KVLD

	while (it_num < 5 && kvldparameters.inlierRate>KVLD(If1, If2,F1,F2, matchesPair, matchesFiltered, vec_score,E,valide,kvldparameters)) {
		kvldparameters.inlierRate/=2;
		std::cout<<"low inlier rate, re-select matches with new rate="<<kvldparameters.inlierRate<<std::endl;
		it_num++;
	}
	std::cout<<"K-VLD filter ends with "<<matchesFiltered.size()<<" selected matches"<<std::endl;
	//=========================================figuring the images============================//
	stringstream f;
	string testid;
	f<<ind1;
	f>>testid;
	f.clear();
	f<<data.I.size();
	string cat;
	f>>cat;

	matchesPair.clear();
	vector<int> result;

	//==============================================learned vector filter==========================================//
	for(int i=0;i< matchesFiltered.size();i++){// it is matchesFiltered
		float error=max(F1[matchesFiltered[i].first].scale,F2[matchesFiltered[i].second].scale) *matchesFiltered[i].weight;
		matchesFiltered[i].weight=error;
	}
	std::sort(matchesFiltered.begin(),matchesFiltered.end(),myrank);
	
	//=============================================end learn and sort vector==================================//
	//======group 1: baseline using RANSAC methods=======//

	for (int j=0;j<items;j++){	
		vector<FCrit> subcrit(iterations); 
		vector<float> mean_N(iterations,0);
		vector<float> mean_e(iterations,0);
#pragma omp parallel
		{	
			srand(int(time(NULL)) ^ (omp_get_thread_num()+1) );
#pragma omp for 
			for (int i=0; i<iterations;i++){

				subcrit[i]=Find_Model_comparison(If1.Width(),If1.Height(),If2.Width(),If2.Height(), F1,F2,
					matchesFiltered,homography,RBmethods[j],OPmethods[j]);
				mean_error(subcrit[i],F1,F2,matchesFiltered,mean_e[i],mean_N[i]);
			}
		}
		float ave_N=std::accumulate(mean_N.begin(),mean_N.end(),0.0)/iterations;
		float ave_e=std::accumulate(mean_e.begin(),mean_e.end(),0.0)/iterations;

		cout<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		 msg<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;

		result.push_back(ave_N);
		  crit.push_back(subcrit);
	}
	cout<<endl;

		
	//======group 2: RANSAC + match selection=======//

	for (int j=0;j<items;j++){
		vector<FCrit> subcrit(iterations); 
		vector<float> mean_N(iterations,0);
		vector<float> mean_e(iterations,0);
		
#pragma omp parallel
		{	
			srand(int(time(NULL)) ^ (omp_get_thread_num()+1));
#pragma omp for  schedule(dynamic,1)
			for (int i=0; i<iterations;i++){
				MatchSelection( If1, If2,	matchesFiltered, F1, F2, mean_e[i], mean_N[i], subcrit[i], homography,RBmethods[j],OPmethods[j]);
			}
		}
		float ave_N=std::accumulate(mean_N.begin(),mean_N.end(),0.0)/iterations;
		float ave_e=std::accumulate(mean_e.begin(),mean_e.end(),0.0)/iterations;

		cout<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		 msg<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		
		result.push_back(ave_N);
		  crit.push_back(subcrit);
	}
	cout<<endl;

	//=============================================correcting matches =========================================================//
	std::vector<libNumerics::matrix<double>> Hlist;	
	Hlist.resize(matchesFiltered.size());

#pragma omp parallel for
	for (int it=0;it<matchesFiltered.size();it++){
		float error= LSFM(chain1,chain2,spline, F1[matchesFiltered[it].first],F2[matchesFiltered[it].second],Hlist[it]);
		
		double x=F1[matchesFiltered[it].first].x, y=F1[matchesFiltered[it].first].y;

		double aX=Hlist[it](0,0)*x+Hlist[it](0,1)*y+Hlist[it](0,2);
		double	aY=Hlist[it](1,0)*x+Hlist[it](1,1)*y+Hlist[it](1,2);
		double	aV=Hlist[it](2,0)*x+Hlist[it](2,1)*y+Hlist[it](2,2);

		double a=pow((Hlist[it](0,0)/aV - (Hlist[it](2,0)*aX)/(aV*aV)),2)
			+pow((Hlist[it](1,0)/aV - (Hlist[it](2,0)*aY)/(aV*aV)),2);

		double b=(Hlist[it](0,0)/aV - (Hlist[it](2,0)*aX)/(aV*aV))
			    *(Hlist[it](0,1)/aV - (Hlist[it](2,1)*aX)/(aV*aV))
			    +(Hlist[it](1,0)/aV - (Hlist[it](2,0)*aY)/(aV*aV))
			    *(Hlist[it](1,1)/aV - (Hlist[it](2,1)*aY)/(aV*aV));

		double c=pow((Hlist[it](0,1)/aV - (Hlist[it](2,1)*aX)/(aV*aV)),2)
			+pow((Hlist[it](1,1)/aV - (Hlist[it](2,1)*aY)/(aV*aV)),2);
		
		double S=sqrt((a-c)*(a-c)+4*b*b)/(a+c); // J^tJ = |a b; b c| 
		matchesFiltered[it].weight=0.3*error+42.6*S;
	}

	std::sort(matchesFiltered.begin(),matchesFiltered.end(),myrank);
	
	//======group 3: RANSAC + match refinement=======//
	for (int j=0;j<items;j++){	
		vector<FCrit> subcrit(iterations); 
		vector<float> mean_N(iterations,0);
		vector<float> mean_e(iterations,0);
#pragma omp parallel
		{	
			srand(int(time(NULL)) ^ (omp_get_thread_num()+1));
#pragma omp for  schedule(dynamic,1)
			for (int i=0; i<iterations;i++){
				subcrit[i]=Find_Model_comparison(If1.Width(),If1.Height(),If2.Width(),If2.Height(),F1,F2,matchesFiltered,homography,RBmethods[j],OPmethods[j]);
				mean_error(subcrit[i],F1,F2,matchesFiltered,mean_e[i],mean_N[i]);
			}
		}
		float ave_N=std::accumulate(mean_N.begin(),mean_N.end(),0.0)/iterations;
		float ave_e=std::accumulate(mean_e.begin(),mean_e.end(),0.0)/iterations;
		cout<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		 msg<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		
		result.push_back(ave_N);
		  crit.push_back(subcrit);
	}
	cout<<endl;
	
    //======group 4: RANSAC + match refinement and match selection=======//
	for (int j=0;j<items;j++){
		vector<FCrit> subcrit(iterations); 
		vector<float> mean_N(iterations,0);
		vector<float> mean_e(iterations,0);
#pragma omp parallel
		{	
			srand(int(time(NULL)) ^ (omp_get_thread_num()+1));
#pragma omp for  schedule(dynamic,1)
			for (int i=0; i<iterations;i++){
				MatchSelection(If1,  If2, matchesFiltered, F1, F2, mean_e[i], mean_N[i], subcrit[i], homography,RBmethods[j],OPmethods[j]);
			}
		}
		float ave_N=std::accumulate(mean_N.begin(),mean_N.end(),0.0)/iterations;
		float ave_e=std::accumulate(mean_e.begin(),mean_e.end(),0.0)/iterations;
		cout<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		 msg<<"Method ID " <<crit.size()<< ", N ratio=" <<float(ave_N)/matchesFiltered.size()<<", N="<<ave_N<<", epipolar error="<<ave_e<<endl;
		
		result.push_back(ave_N);
		  crit.push_back(subcrit);
	}
	cout<<endl;

	//=====================================================================//
	 msg<<endl;
	delete[] I1.data;
	delete[] I2.data;
	return result;
}

//========================================== plan test======================================================// 

int main(int argc,char*argv[]) {
	//==========we take a pair of strecha dataset as example.
	bool have_groundtruth=true; 
	int numbers = 8;//number of images
	int iterations=16;// number of iterations
	
	string Path=std::string(THIS_SOURCE_DIR);
	string src=Path+"/Input/herzjesu_dense/";
	string out=Path+"/Output/herzjesu_dense/";

	cout<< "Warming:"<<endl
		<< "1.The comparison test is time consuming "<<endl
	    << "2.It has been developed under Windows, may have problems with other OS."<<endl;
	cout<< "Test information: "<<endl
		<<"1. Check if result is already in the output folder"<<endl
		<<"2. In MRMS.cpp use '#pragma omp for' around the line 771 to accelerate "<<endl
		<<"3. the test runs for setting : "<<endl
		<< "       "<<numbers-1 << " image pairs "<<endl
		<< "       "<< 4 <<" method as ORSA, MS, MR, MR+MS "<<endl
		<< "       "<< iterations<< " iterations for each method" << endl
		<<"======================================================="<<endl;

	Database Strecha(src,numbers);

	//==============================generate features===========================//
#pragma omp parallel for
	for (int i =0; i<Strecha.size; i++){
		cv::PyramidAdaptedFeatureDetector detector2(new  cv::SiftFeatureDetector(),5);// 5 levels of image scale
		detector2.detect(Strecha.I[i],Strecha.keys[i]);
		std::cout<< "sift::image "<< i<<" with " << Strecha.keys[i].size() <<" keypoints" <<std::endl;
		cv::SiftDescriptorExtractor extractor;
		extractor.compute(Strecha.I[i],Strecha.keys[i],Strecha.descriptors[i]);	
	}

	//===================================== methode evaluation ======================================//
	
	vector<vector<float>> errorR(numbers-1);// error[a][b] is the result for pair ID a,  method ID b.
	vector<vector<float>> errorT(numbers-1);
	vector<vector<int>> Ns(numbers-1,0);

	
	for (int i =0; i<Strecha.size-1; i++){
		int j= (i+1)%Strecha.size;
		cout<<"==============test with paire "<<i<<" and "<<j<<" =========="<<endl;
		std::stringstream f;
		f<<i;
		string index1;
		f>>index1;
		f.clear();
		f<<j;
		string index2;
		f>>index2;
		bool homography=false;
		//===================== build ground truth ==================//  
		cv::Mat trueR= Strecha.R[j]*Strecha.R[i].t();
		cv::Mat trueT=-Strecha.R[j]*(-Strecha.R[j].t()*Strecha.T[j]+Strecha.R[i].t()*Strecha.T[i]);
		double normT=cv::norm(trueT);
		double* tR=(double*) trueR.data;
		double* tT=(double*) trueT.data;

		std::ofstream matlab(out+index1+"_"+index2+"_trueP.txt");
		matlab<<tR[0*3+0]<<" "<<tR[0*3+1]<<" "<<tR[0*3+2]<<" "<<tT[0]<<endl
			<<tR[1*3+0]<<" "<<tR[1*3+1]<<" "<<tR[1*3+2]<<" "<<tT[1]<<endl
			<<tR[2*3+0]<<" "<<tR[2*3+1]<<" "<<tR[2*3+2]<<" "<<tT[2]<<endl;
		matlab.close();
		//========= Computing F matrices by various methods ==============//
		ofstream logfile(out+index1+"_"+index2+"_log.txt");

		//F estimations are realized by the following two lines
		vector<vector<FCrit>> crit; // crit[a][b] is the result for method ID a,  iteration ID b.
		Ns[i]=Comparing(Strecha,i, j,crit, homography,iterations, logfile);
		//=============== Evaluate F matrices ==============//
		errorR[i].resize(crit.size(),0);
		errorT[i].resize(crit.size(),0);
		
		int methodID=0;
		
		for(int id=0;id<crit.size();id++){
			vector<float> vT;
			vector<float> vR;
			for(int k=0 ; k < iterations; k++){
				//===passe to essential matrix===//
				cv::Mat fonda1=convert(crit[id][k].F);
				cv::Mat ess1=Strecha.K[j].t()*fonda1*Strecha.K[i];
				cv::Mat Winv1(3,3,cv::DataType<double>::type);
				double* Wptr1=(double*) Winv1.data;
				for (int it= 0; it< 9 ; it++) Wptr1[it]=0;
				Wptr1[0*3+1]=1;  Wptr1[1*3]=-1; Wptr1[2*3+2]=1; 

				//===SVD decomposition===//
				cv::SVD svd(ess1,cv::SVD::MODIFY_A);
				cv::Mat S1=svd.w, U1=svd.u,Vt1=svd.vt;
				

				//===estimate rotation===//
				cv::Mat R1=U1*(Winv1*Vt1);
				if(cv::determinant(R1)<0) R1=-R1;
				double* Rptr1=(double*) R1.data;

				if ((Rptr1[0]+Rptr1[4]+Rptr1[8]-1)<0){ 
					R1=U1*Winv1.t()*Vt1;
					if(cv::determinant(R1)<0) {
						R1=-R1;
					}
				}
				Rptr1=(double*) R1.data;
				
				//==estimate translation==//
				Wptr1[2*3+2]=0;
				cv::Mat Tx1=-Vt1.t()*(Winv1*(-1))*Vt1;
				double* Txptr1=(double*) Tx1.data;
				cv::Mat T1(3,1,cv::DataType<double>::type);

				double* Tptr1=(double*) T1.data;
				Tptr1[0]=Txptr1[2*3+1]; Tptr1[1]=Txptr1[0*3+2]; Tptr1[2]=Txptr1[1*3+0];
				T1=R1*T1*normT;Tptr1=(double*) T1.data;
				
				cv::Mat s=T1.t()*trueT;
				double* sptr=(double*) s.data;
				if (sptr[0]<0){
					T1=-T1;
					Tptr1=(double*) T1.data;
				}
				//===============errors=====================//
				cv::Mat E= R1.t()*trueR;
				E=(E.t()-E)/2;
				double* eptr=(double*) E.data;
				double eR=asin(sqrt(eptr[0*3+1]*eptr[0*3+1]+eptr[0*3+2]*eptr[0*3+2]+eptr[1*3+2]*eptr[1*3+2]));
				double eT=cv::norm(T1-trueT)/cv::norm(trueT);
				//===============writing=====================//
				errorR[i][id]+=eR*eR;
				errorT[i][id]+=eT*eT;
				
			}
			//===============put values into the vectors===================//
			errorR[i][id]=sqrt(errorR[i][id]/iterations);
			errorT[i][id]=sqrt(errorT[i][id]/iterations);
			//==============writing output files
			std::stringstream fid;
			fid<<methodID;
			string idstr;
			fid>>idstr;

			logfile<<"method ID "<<idstr<<" R error "<<errorR[i][id]<<" t error"<<errorT[i][id]<<" number "<<Ns[i][id]<<endl;
			   cout<<"method ID "<<idstr<<" R error "<<errorR[i][id]<<" t error"<<errorT[i][id]<<" number "<<Ns[i][id]<<endl;

			methodID++;
			if ((id+1)%(5)==0){
				cout<<endl;
				logfile<<endl;
			}
		}
		logfile.close();
	}

	ofstream resultfile(out+"final_average_result.txt");
	for(int id=0;id<errorR[0].size();id++){
		double eR=0;
		double et=0;
		for (int i =0; i<Strecha.size-1; i++){
			int j= (i+1)%Strecha.size;
			//cout<<"==============test with paire "<<i<<" and "<<j<<" =========="<<endl;
			std::stringstream f;
			f<<i;
			string index1;
			f>>index1;
			f.clear();
			f<<j;
			string index2;
			f>>index2;
			eR+=errorR[i][id];
			et+=errorT[i][id];
		}
		std::stringstream fid;
		fid<<id;
		string idstr;
		fid>>idstr;
		resultfile<<"method ID "<<idstr<<" R error "<<eR/errorR.size()<<" t error"<<et/errorT.size()<<endl;
	}
	resultfile.close();
	//==========================write files==================================//
	ofstream info(out+"rotation_info.txt");
	for (int i =0; i<Strecha.size-1; i++){
		int j= (i+1)%Strecha.size;
		cv::Mat trueR= Strecha.R[j]*Strecha.R[i].t();
		cv::Mat trueT=-Strecha.R[j]*(-Strecha.R[j].t()*Strecha.T[j]+Strecha.R[i].t()*Strecha.T[i]);
		double normT=cv::norm(trueT);
		double* tR=(double*) trueR.data;
		double* tT=(double*) trueT.data;
		info<<asin(sqrt(tR[0*3+1]*tR[0*3+1]+tR[0*3+2]*tR[0*3+2]+tR[1*3+2]*tR[1*3+2]))<<" ";
	}
	info.close();
	
	return 0;
}