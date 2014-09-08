/** @Converting functions between structures of openCV and KVLD suitable structures
 ** @author Zhe Liu
 **/
/*
Copyright (C) 2007-12 Zhe Liu
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include "MRMS.h"

//======================================database setting==================================================//
	Database::~Database(){
			size=0;
	}
	Database::Database(int s){
		size=s;
		name.resize(size);
		I.resize(size);
		K.resize(size);
		P.resize(size);
		R.resize(size);
		T.resize(size);
		keys.resize(size);
		descriptors.resize(size);	
		Icolor.resize(size);
	}
	Database::Database(string path,int number){
		size=number;
		cout<<"importing "<<size<<" images"<<endl;
		keys.resize(size);
		descriptors.resize(size);

			for(int i=0; i<number; i++){
				string unused;
				stringstream f;
				f.clear();
				f<<i;
				string s;
				f>>s;
				
				if (i<10) name.push_back(path+"000"+s+ ".png");
				else name.push_back(path+"00"+s+ ".png");

				cout<<"'"<<name[i]<<"'"<<endl;
				I.push_back(cv::imread(name[i], CV_LOAD_IMAGE_GRAYSCALE));
				Icolor.push_back(cv::imread(name[i], CV_LOAD_IMAGE_COLOR));
				ifstream str(name[i]+".camera");

				cv::Mat k(3,3,cv::DataType<double>::type);

				double* kptr=(double*)k.data;

				str>>kptr[0*3+0]>>kptr[0*3+1]>>kptr[0*3+2]
				>>kptr[1*3+0]>>kptr[1*3+1]>>kptr[1*3+2]
				>>kptr[2*3+0]>>kptr[2*3+1]>>kptr[2*3+2];
				K.push_back(k);

				str>>unused>>unused>>unused;

				//==============Rotation ==========//
				//Caution: it's the translation of rotation

				cv::Mat r(3,3,cv::DataType<double>::type);
				double* rptr=(double*)r.data;

				str>>rptr[0*3+0]>>rptr[0*3+1]>>rptr[0*3+2]
				>>rptr[1*3+0]>>rptr[1*3+1]>>rptr[1*3+2]
				>>rptr[2*3+0]>>rptr[2*3+1]>>rptr[2*3+2];
				r=r.t();
				R.push_back(r);

				//==============Translation ==========//
				//Caution: Translation = -RC we read C in the file.

				cv::Mat c(3,1,cv::DataType<double>::type);
				double* cptr=(double*)c.data;
				str>>cptr[0]>>cptr[1]>>cptr[2];
				cv::Mat t=-r*c;
				T.push_back(t);

				str.close();



				cv::Mat p(3,4,cv::DataType<double>::type);
				double* pptr=(double*)p.data;
				ifstream str2(name[i]+".P");
				str2>>pptr[0*4+0]>>pptr[0*4+1]>>pptr[0*4+2]>>pptr[0*4+3]
				>>pptr[1*4+0]>>pptr[1*4+1]>>pptr[1*4+2]>>pptr[1*4+3]
				>>pptr[2*4+0]>>pptr[2*4+1]>>pptr[2*4+2]>>pptr[2*4+3];
				P.push_back(p);
				str2.close();

			}
		cout<<"database imported"<<endl;
	}

//=======================================image scales===============================================//
/*
We construct the image scales by bluring without subsampling at each level. 
*/

ImageChain::ImageChain(const cv::Mat& I,int spline): spline(spline){
		w=I.cols;
		h=I.rows;
		LWImage<float> img=Convert_image(I);
		prepare_spline(img,spline);
		map.push_back(img);

		float sig=1.6f;
		float fact=1.0f;
		scales.push_back(fact);
		
		for (int i= 0; i<4; i++){
			float sigma=fact*sig; 
			cv::Mat Ib;
			cv::GaussianBlur(I,Ib,cv::Size(0,0),sigma,sigma);
			LWImage<float> imgb=Convert_image(Ib);
			prepare_spline(imgb,spline);
			map.push_back(imgb);
			scales.push_back(2.0*fact);
			fact*=2.0;
		}
	}
ImageChain::~ImageChain(){
		for (int i= 0; i<map.size(); i++){
			delete[] map[i].data;
		}
	}
const LWImage<float>& ImageChain::operator[](size_t ind)const{
		return map[ind];
	}
size_t ImageChain::size()const{
		return map.size();
	}

//======================================Match Refinemetion===========================================//
  Cell::Cell(const LWImage<float>& I,const keypoint& key,int sp,float dim,bool norm , float deform,float sigma): norm(norm),deform(deform),sigma(sigma){
	 
	  initial(I,key.x,key.y,key.scale,key.angle,sp,dim);
  }

  void Cell::creat(const LWImage<float>& I,const keypoint& key,int sp,float dim,bool norm , float deform,float sigma){
	
	this->norm=norm;
	this->deform=deform;
	this->sigma=sigma;
	initial(I,key.x,key.y,key.scale,key.angle,sp,dim);
  }

  void Cell::updateP1(const LWImage<float>& I,float dim){
	  assert(sigma>=0);
	  initial(I,pt[0],pt[1],pt[2],pt[3],spline,dim);
  }

  void Cell::initial(const LWImage<float>& I,float x,float y, float scale,float angle,int sp,float dim){
	  //===================initializing storage===========//

	  if (posi.size()==0){
		  posi.resize((size+1)/2);
		  float ratio=1.0+deform;
		  float step=1.0f;
		  posi[0]=0;
		  for (int i=1; i<=(size-1)/2; i++) {
			  posi[i]=posi[i-1]+step;
			  step*=ratio;
		  }   			   
	  }
	  if (weight.size()==0){
		  weight.resize((size+1)/2);
		  if (sigma>0){
			  for (int i=0; i<=(size-1)/2; i++) {
				  weight[i]= exp(-pow(
					  posi[i]
				  /(posi[(size-1)/2]*sigma),
					  2)
					  /2
					  );  
			  }
		  }else{
			  for (int i=0; i<=(size-1)/2; i++) {
				  weight[i]= 1;  
			  }
		  }
	  }

	  spline=sp;
	  //=========coordinate
	  pt[0]=x;
	  pt[1]=y;
	  pt[2]=scale;
	  pt[3]=angle;

	  N=20;
	  real_step=scale*dim*2 /(size-1);
	  Va[0]= std::cos(angle)*real_step/N;
	  Va[1]= std::sin(angle)*real_step/N;
	  Vb[0]=-std::sin(angle)*real_step/N;
	  Vb[1]= std::cos(angle)*real_step/N;


	  float* indF=F;
	  float* indG=G;


	  //=========initial values
	  for (int i=-(size-1)/2; i<=(size-1)/2;i++){
		  float signi=(i>0)? 1:-1; 
		  for (int j=-(size-1)/2; j<=(size-1)/2; j++){
			  float signj=(j>0)? 1:-1;
			  float i_x=pt[0]+N*(signi*Va[0]*posi[abs(i)]+signj*Vb[0]*posi[abs(j)]);
			  float i_y=pt[1]+N*(signi*Va[1]*posi[abs(i)]+signj*Vb[1]*posi[abs(j)]);
			  int a=i+(size-1)/2,b=j+(size-1)/2;
			  //==========0 and 1st order=========//
			  if (i_x<1||i_x>I.w-1 || i_y<1||i_y>I.h-1){
				  *(indF++)=0.0f;
				  *(indG++)=0;
				  *(indG++)=0;
			  }else{
				  *(indF++)=get(I,spline,i_x,i_y);
				  *(indG++)=N*(get(I,spline,i_x+real_step, i_y          )-get(I,spline,i_x-real_step, i_y          ))/(2*real_step);
				  *(indG++)=N*(get(I,spline,i_x,           i_y+real_step)-get(I,spline,i_x,           i_y-real_step))/(2*real_step);
			  }
		  }
	  }
	  if (norm) normalize();
	  else {
		  var=0;
		  rs=0;
		  rt=0;
	  }
  }

  CoupleM::CoupleM(bool normalize,float deform, float sigma):normalize(normalize), deform(deform),sigma(sigma){}

  template<typename T>
  CoupleM::CoupleM(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,const int spline, const float dim, 
	  bool normalize, float deform,float sigma):normalize(normalize), deform(deform),sigma(sigma){

		  initial(I1,key1,I2,key2,spline,dim);
  }

	template<typename T>
	void CoupleM::initial(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,const int spline, const float dim){
		size=(float) Cell::size;
		float mid_ratio=float(size-1)/2/min(key1.scale,key2.scale);
		
		dimension=dim*mid_ratio;
		scale_ratio=key2.scale/key1.scale;

		P1.creat(I1,key1,spline,dimension,normalize,deform,sigma);
		P2.creat(I2,key2,spline,dimension,normalize,deform,sigma);
		if (!normalize) P2.findR(P1);
		
		float angle=key2.angle-key1.angle;
		
		float ratio=P1.N*key2.scale/(key1.scale*P2.N);

		float a=std::cos(angle)*ratio, b=std::sin(angle)*ratio;
		M[0]= a;
		M[1]=-b;
		M[2]=0;

		M[3]= b;
		M[4]= a;
		M[5]=0;
		M[6]=0;
		M[7]=0;
	}
	
	template<typename T>
	void CoupleM::update(const T& I1,const keypoint& key1,const T& I2,const keypoint& key2,float dim){
		float mid_ratio=float(size-1)/2/min(key1.scale,key2.scale);
		dimension=dim*mid_ratio;
		
			P1.updateP1(I1,dimension);
			P2.ModelP2(P1,I2,scale_ratio,M);
		
	}
	
	template<typename T>
	void  CoupleM::ModelP2(const T& I2){
		
			P2.ModelP2(P1,I2,scale_ratio,M);
	}

	//================local homography matrix OI======//
	void CoupleM::loadH(const libNumerics::matrix<double>& H){
		libNumerics::matrix<double> t1=libNumerics::matrix<double>::eye(3); 
		libNumerics::matrix<double>	s1=libNumerics::matrix<double>::eye(3);
		libNumerics::matrix<double>	negt2=libNumerics::matrix<double>::eye(3);
		libNumerics::matrix<double> negs2=libNumerics::matrix<double>::eye(3);
		
		t1(0,2)=P1.pt[0];
		t1(1,2)=P1.pt[1];
		s1(0,0)=P1.N;
		s1(1,1)=P1.N;

		negt2(0,2)=-P2.pt[0];
		negt2(1,2)=-P2.pt[1];
		negs2(0,0)=1.0/P2.N;
		negs2(1,1)=1.0/P2.N;
		libNumerics::matrix<double> Ht=negs2*negt2*H*t1*s1;

		M[0]=Ht(0,0)/Ht(2,2);
		M[1]=Ht(0,1)/Ht(2,2);
		M[2]=Ht(0,2)/Ht(2,2);
		M[3]=Ht(1,0)/Ht(2,2);
		M[4]=Ht(1,1)/Ht(2,2);
		M[5]=Ht(1,2)/Ht(2,2);
		M[6]=Ht(2,0)/Ht(2,2);
		M[7]=Ht(2,1)/Ht(2,2);
	}
	libNumerics::matrix<double> CoupleM::toMatrix(){

		libNumerics::matrix<double> Ht;
		Ht(0,0)=M[0];
		Ht(0,1)=M[1];
		Ht(0,2)=M[2];
		Ht(1,0)=M[3];
		Ht(1,1)=M[4];
		Ht(1,2)=M[5];
		Ht(2,0)=M[6];
		Ht(2,1)=M[7];
		Ht(2,2)=1;
		
		libNumerics::matrix<double> negt1=libNumerics::matrix<double>::eye(3),   negs1=libNumerics::matrix<double>::eye(3),
			t2=libNumerics::matrix<double>::eye(3), s2=libNumerics::matrix<double>::eye(3);
		negt1(0,2)=-P1.pt[0];
		negt1(1,2)=-P1.pt[1];
		negs1(0,0)=1.0/P1.N;
		negs1(1,1)=1.0/P1.N;

		t2(0,2)=P2.pt[0];
		t2(1,2)=P2.pt[1];
		s2(0,0)=P2.N;
		s2(1,1)=P2.N;
		libNumerics::matrix<double> H=t2*s2*Ht*negs1*negt1;
		return H;
	}
	void CoupleM::loadH(double H[8]){
		for (int i=0; i<8; i++)	M[i]=H[i];
	}	
	void CoupleM::toMatrix(double H[8]){
		H[0]=M[0]; 	H[1]=M[1];	H[2]=M[2];	
		H[3]=M[3];	H[4]=M[4];	H[5]=M[5];
		H[6]=M[6];	H[7]=M[7];
	}
	float CoupleM::diff(){
		float error=0;
		
			assert(P2.var>0 || P2.rs>0);
			if ( normalize==false) P2.findR(P1);
			error+=P1.diff(P2);
		
		return error;
	}	
	void CoupleM::estimate_lsm(float dM[8]){
		assert(normalize==false);

		int size=P1.size;
		int cols=6;
		cv::Mat W(size*size,cols,cv::DataType<float>::type);
		float* ptrW = (float*) W.data;
		cv::Mat T(size*size,1,cv::DataType<float>::type);
		float* ptrT = (float*) T.data;
		
			P2.findR(P1);
			for (int i=-(size-1)/2; i<=(size-1)/2;i++){
				float signi=(i>0)? 1:-1;
				for (int j=-(size-1)/2; j<=(size-1)/2; j++){
					float signj=(j>0)? 1:-1;
					float i_x=signi*P1.Va[0]*P1.posi[abs(i)]+signj*P1.Vb[0]*P1.posi[abs(j)];
					float i_y=signi*P1.Va[1]*P1.posi[abs(i)]+signj*P1.Vb[1]*P1.posi[abs(j)];
					float w=P1.weight[abs(i)]*P1.weight[abs(j)];
					int a=i+(size-1)/2,b=j+(size-1)/2;
					
					int indexT=(a*size+b);
					int indexW=indexT*cols;
			
					ptrW[indexW+0]=w* i_x*P2.G[2*indexT];
					ptrW[indexW+1]=w* i_y*P2.G[2*indexT];
					ptrW[indexW+2]=w*     P2.G[2*indexT];

					ptrW[indexW+3]=w* i_x*P2.G[2*indexT+1];
					ptrW[indexW+4]=w* i_y*P2.G[2*indexT+1];
					ptrW[indexW+5]=w*     P2.G[2*indexT+1];

					ptrT[indexT]=w*(P1.F[indexT]-P2.rs*P2.F[indexT]-P2.rt)/P2.rs;

				}
			}
		
		cv::Mat d=(W.t()*W).inv()*W.t()*T;
		float* ptrD = (float*) d.data;
		for (int i=0; i<cols; i++){
			dM[i]=ptrD[i];
		}
		dM[6]=0;
		dM[7]=0;
	}	
	void CoupleM::estimateAffine(float dM[8]){
		int size=P1.size;
		int cols=6;
		cv::Mat W(size*size,cols,cv::DataType<float>::type);
		float* ptrW = (float*) W.data;
		cv::Mat T(size*size,1,cv::DataType<float>::type);
		float* ptrT = (float*) T.data;

	
			if (!normalize) P2.findR(P1);
			for (int i=-(size-1)/2; i<=(size-1)/2;i++){
				float signi=(i>0)? 1:-1;
				for (int j=-(size-1)/2; j<=(size-1)/2; j++){
					float signj=(j>0)? 1:-1;
					float i_x=signi*P1.Va[0]*P1.posi[abs(i)]+signj*P1.Vb[0]*P1.posi[abs(j)];
					float i_y=signi*P1.Va[1]*P1.posi[abs(i)]+signj*P1.Vb[1]*P1.posi[abs(j)];
					float w=P1.weight[abs(i)]*P1.weight[abs(j)];
					int a=i+(size-1)/2,b=j+(size-1)/2;
					
					int indexT=(a*size+b);
					int indexW=indexT*cols;
					
					float norm=1+M[6]*i_x+M[7]*i_y;

					ptrW[indexW+0]=w* i_x*P2.G[2*indexT]/norm;
					ptrW[indexW+1]=w* i_y*P2.G[2*indexT]/norm;
					ptrW[indexW+2]=w*     P2.G[2*indexT]/norm;

					ptrW[indexW+3]=w* i_x*P2.G[2*indexT+1]/norm;
					ptrW[indexW+4]=w* i_y*P2.G[2*indexT+1]/norm;
					ptrW[indexW+5]=w*     P2.G[2*indexT+1]/norm;

					if (normalize)  
						ptrT[indexT]=w*(P1.F[indexT]-P2.F[indexT])*P2.var/(1-(1.0+P2.F[indexT]*P2.F[indexT])/(size*size));
					else 
						ptrT[indexT]=w*(P1.F[indexT]-P2.rs*P2.F[indexT]-P2.rt)/P2.rs;
				}
			}
	
		cv::Mat d=(W.t()*W).inv()*W.t()*T;
		float* ptrD = (float*) d.data;
		for (int i=0; i<cols; i++){
			dM[i]=ptrD[i];
		}
		dM[6]=0;
		dM[7]=0;
	}
	void CoupleM::output(float & x, float & y){
		float x1,y1;
		locate(0.0f,0.0f,M,x1,y1);
		x=P2.pt[0]+x1*P2.N;
		y=P2.pt[1]+y1*P2.N;
	}

/*
LSM methods for comparison
*/
float LSM(const LWImage<float>& I1, const LWImage<float>& I2,const int spline, const keypoint& key1, keypoint& key2)
{
	float sigma=0;
	float deform=0;
	float resize=1.0f;
	bool norm=false;

	CoupleM couple(I1,key1,I2,key2,spline,resize,norm,deform,sigma);

	int local_min=0;
	float local_thres=0.98; //the error is N2 without sqrt,so the value is about (0.95)^2
	//size_t max_time=3;
	float Bv=couple.diff();
	if(    key1.x>key1.scale*1.5*couple.dimension && I1.w-key1.x>key1.scale*1.5*couple.dimension
		&& key1.y>key1.scale*1.5*couple.dimension && I1.h-key1.y>key1.scale*1.5*couple.dimension
		&& key2.x>key2.scale*1.5*couple.dimension && I2.w-key2.x>key2.scale*1.5*couple.dimension
		&& key2.y>key2.scale*1.5*couple.dimension && I2.h-key2.y>key2.scale*1.5*couple.dimension
		)
	{
		for (int i=0; i<80; i++){	
			float local_bv=Bv;
			//=========affine part==========//
			float dM[8];

			if (!norm) couple.estimate_lsm(dM);
			else couple.estimateAffine(dM);

			size_t it=0;
			add(couple.M,dM);
			couple.ModelP2(I2);
			float v= couple.diff();
			if (v<Bv){
				Bv=v; break;
			}else{
				moins(couple.M,dM);
				couple.ModelP2(I2);
			}
			//======breaking step=======//
			if (local_thres<Bv/local_bv){local_min++;}
			else{local_min=0;}

			if (local_min==6) break;
		}
		couple.output(key2.x,key2.y);
	}
	return Bv;
}

/*
LSFM method, our extension of LSM
*/
float LSFM(const ImageChain& I1, const ImageChain& I2,const int spline, const keypoint& key1,keypoint& key2, 
	libNumerics::matrix<double>& H)
{	
	float deform=0.1;// rot = 1+deform
	float resize=1.571f;
	float sigma=0.9;/*for Gaussian weight*/

	bool scale=true;
	bool norm=true;

	size_t max_time=3;
	float local_thres=0.98;
	CoupleM couple(norm,deform,sigma);
	bool first=true;

	int start_level=I1.size()-1; 	
	if (!scale) start_level==0;

	int index=0;
	float min_val=-1;
	for (int i=start_level;i>=0; i--){
		CoupleM couple2(I1[index],key1,I2[index],key2,spline,resize*I1.scales[index],norm,deform,sigma);
		float val2=couple2.diff();
		if (min_val==-1 || val2<min_val){
			min_val=val2;
			couple=couple2;
			index=i;
		}
	}

	for (;index>=0;index--){
		if(first){ first=false;}
		else {

			CoupleM couple2(I1[index],key1,I2[index],key2,spline,resize*I1.scales[index],norm,deform,sigma);
			float val1=couple2.diff();
			couple.update(I1[index],key1,I2[index],key2,resize*I1.scales[index]);
			float val2=couple.diff();
			if (val1<val2) couple=couple2;
		}
		int local_min=0;
		float Bv=couple.diff();

		if(    key1.x>key1.scale*1.5*couple.dimension && I1.w-key1.x>key1.scale*1.5*couple.dimension
			&& key1.y>key1.scale*1.5*couple.dimension && I1.h-key1.y>key1.scale*1.5*couple.dimension
			&& key2.x>key2.scale*1.5*couple.dimension && I2.w-key2.x>key2.scale*1.5*couple.dimension
			&& key2.y>key2.scale*1.5*couple.dimension && I2.h-key2.y>key2.scale*1.5*couple.dimension
			)
		{
			for (int i=0; i<80; i++){	
				float local_bv=Bv;

				//=========affine part==========//
				float dM[8];
				couple.estimateAffine(dM);

				size_t it=0;
				do{
					add(couple.M,dM);
					couple.ModelP2(I2[index]);
					float v= couple.diff();
					if (v<Bv){
						Bv=v; break;
					}else{
						moins(couple.M,dM);
						for (int i= 0; i<8;i++) dM[i]/=2;
						it++;
						if(it==max_time) couple.ModelP2(I2[index]);
					}
				}while( it<max_time);

				//======breaking step=======//
				if (local_thres<Bv/local_bv){local_min++;}
				else{local_min=0;}
				if (local_min==6) break;
			}
		}
	}
	couple.output(key2.x,key2.y);
	H=couple.toMatrix();
	return couple.diff();
}

void get_RT(const FCrit& crit, const cv::Mat& K1,const cv::Mat& K2, double norm, cv::Mat& R,cv::Mat& T){
	cv::Mat fonda=convert(crit.F);
	cv::Mat ess=K2.t()*fonda*K1;
	//===SVD===//
	cv::Mat S, U,Vt,Tx;
	cv::SVD::compute(ess,S,U,Vt);
	cv::Mat Winv(3,3,cv::DataType<double>::type);
	double* Wptr=(double*) Winv.data;
	Wptr[0*3+1]=1;  Wptr[1*3]=-1; Wptr[2*3+2]=1;

	//===rotation===//
	R=U*Winv*Vt;
	if(cv::determinant(R)<0) R=-R;

	double* Rptr=(double*) R.data;

	if ((Rptr[0]+Rptr[4]+Rptr[8]-1)<0){ 
		R=U*Winv.t()*Vt;
		if(cv::determinant(R)<0) {
			R=-R;
			Rptr=(double*) R.data;
		}
	}
	//==translation==//
	Wptr[2*3+2]=0;
	Tx=-Vt.t()*(Winv*(-1))*Vt;
	double* Txptr=(double*) Tx.data;

	double* Tptr=(double*) T.data;
	Tptr[0]=Txptr[2*3+1]; Tptr[1]=Txptr[0*3+2]; Tptr[2]=Txptr[1*3+0];
	T=R*T*norm;Tptr=(double*) T.data;
}

//==================================== Point distribution==========================================//
void Matcher(const cv::Mat& descriptors1, const cv::Mat& descriptors2,std::vector<cv::DMatch>& matches,
	const float matching_criterion, const bool bSymmetricMatches, vector<float>& loweScore){
	matches.clear();
	cv::BFMatcher matcher(cv::NORM_L2);
	std::vector<std::vector<cv::DMatch>> knnmatches;
	matcher.knnMatch(descriptors1,descriptors2,knnmatches,2);

	if (!bSymmetricMatches){
		for (std::vector<std::vector<cv::DMatch>>::const_iterator it=knnmatches.begin();it!=knnmatches.end();it++){
			if (it->at(0).distance<matching_criterion*it->at(1).distance) {
				matches.push_back((*it)[0]);
				loweScore.push_back( it->at(0).distance/it->at(1).distance);
			}
		}
	}else{
		vector<int>	M2(descriptors2.rows);
		vector<float> M2score(descriptors2.rows);
		
		std::vector<std::vector<cv::DMatch>> knnmatches2;
		matcher.knnMatch(descriptors2,descriptors1,knnmatches2,2);

		for (std::vector<std::vector<cv::DMatch>>::const_iterator it=knnmatches2.begin();it!=knnmatches2.end();it++){
			if (it->at(0).distance<matching_criterion*it->at(1).distance){
				M2[it->at(0).queryIdx]=it->at(0).trainIdx;
				M2score[it->at(0).queryIdx]=it->at(0).distance/it->at(1).distance;
			}else{
				M2[it->at(0).queryIdx]=-1;
				M2score[it->at(0).queryIdx]=-1;
			}
		}

		for (std::vector<std::vector<cv::DMatch>>::const_iterator it=knnmatches.begin();it!=knnmatches.end();it++){
			if (it->at(0).distance<matching_criterion*it->at(1).distance 
				&& M2[it->at(0).trainIdx]==it->at(0).queryIdx){
				matches.push_back((*it)[0]);
				loweScore.push_back(max( it->at(0).distance/it->at(1).distance, M2score[it->at(0).trainIdx]));
			}
		}
	}
}


//======================================Match Selection===========================================//
void  Criterion(const FCrit& crit, int w, int h,  const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,const std::vector<Pair>& matches,float& e){
	e=0;
	int N=0;
	int size= 9;
	if(crit.thres<=0) return;
	for(vector<Pair>::const_iterator it=matches.begin();it!=matches.end();it++){
		float error=crit.error(F1[it->first],F2[it->second]);
		if (error<=crit.thres){
			N++;
		}
	}
    if (N<size) return;
	
	double scale= 0.5*sqrt((double)w*h);
	cv::Mat mat(N,size,cv::DataType<double>::type);
	double* ptr=(double*) mat.data;
	//int i=0;
	for(vector<Pair>::const_iterator it=matches.begin();it!=matches.end();it++){
		float error=crit.error(F1[it->first],F2[it->second]);
		if (error<=crit.thres){
			double x1=F1[it->first ].x/scale, y1=F1[it->first ].y/scale, x2=F2[it->second].x/scale, y2=F2[it->second].y/scale;
			*ptr++=x1*x2;
			*ptr++=y1*x2;
			*ptr++=x2;
			*ptr++=y2*x1;
			*ptr++=y2*y1;
			*ptr++=y2;
			*ptr++=x1;
			*ptr++=y1;
			*ptr++=1;
		}
	}
	cv::PCA pca(mat,cv::noArray(),CV_CALIB_USE_INTRINSIC_GUESS);
	cv::Mat values= pca.eigenvalues;
	double* vptr=(double*) values.data;
	e=vptr[7];	
}

void  mean_error(const FCrit& crit, const std::vector<keypoint>& F1,const std::vector<keypoint>& F2,const std::vector<Pair>& matches, float & e, float& cardinal,int OPmethod){
	e=0;
	cardinal=0;
	if (crit.thres<=0) return;
	for(vector<Pair>::const_iterator it=matches.begin();it!=matches.end();it++){
		float error=crit.error(F1[it->first],F2[it->second]);
		if (error<=crit.thres){
			e+=error*error;
			cardinal+=1;
		}
	}
	if (cardinal<8) {
		e=0;
		return;
	}
	e=sqrt(e/(cardinal-7));
	
	
	if (OPmethod==2)//M-estimator
	{
		float eM=0,cardM=0;
		for(vector<Pair>::const_iterator it=matches.begin();it!=matches.end();it++){
			float value=crit.error(F1[it->first],F2[it->second]);
			if (value<e){
				eM+=pow(value,2);
				cardM++;
			}else if (abs(value)<3*e){
				eM+=abs(value)*e;
				cardM+=e/abs(value);
			}					
		}
		e=sqrt(eM/cardM);
		cardinal=cardM;	
	}
}

void MatchSelection(const Image<float>& If1, const Image<float>& If2,
	const std::vector<Pair>& matchesSorted, const std::vector<keypoint>& F1,std::vector<keypoint>& F2,
	float& b_e, float& b_N, FCrit& b_c, bool homography, int RBmethod,int OPmethod, float rate,bool second_check){
		const size_t steps=13;
		float ratio[steps]={0.4,0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};
		
		vector<float> l_N(steps,0);
		vector<float> l_e(steps,0);
		vector<FCrit> l_c(steps);

		if (matchesSorted.empty()) {
			b_N=0;
			b_e=0;
			return;
		}
		std::vector<std::vector<Pair>> lists(steps);
		std::vector<float> indicators(steps,0);
		//=============================generate protential results===================//
		for(int j =0; j<steps;j++){
			std::vector<Pair> Corrected(matchesSorted.begin(),matchesSorted.begin()+int(ratio[j]*matchesSorted.size()));
			float scale= Corrected[Corrected.size()-1].weight*Corrected[Corrected.size()-1].weight;
			l_c[j]=Find_Model_comparison(If1.Width(),If1.Height(),If2.Width(),If2.Height(),	F1,F2,Corrected,homography,RBmethod,OPmethod);
			mean_error(l_c[j],F1,F2,Corrected,l_e[j],l_N[j],OPmethod);
			lists[j]=Corrected;
			//============================robustness of the result=====================//
			Criterion(l_c[j], If1.Width(),If1.Height(), F1,F2,Corrected, indicators[j]);
			//cout<<l_e[j]<<" "<<l_N[j]<<" "<<indicators[j]<<" "<<l_c[j].thres<<endl;
		}
		float indica_max= *std::max_element(indicators.begin(),indicators.end());
		float stable=60; 


		

		int initial=0;
		int b_i=initial;
		b_e=l_e[initial];
		b_N=l_N[initial];
		b_c=l_c[initial];
		while((!_finite(b_e)|| b_e<=0||!_finite(b_N)||indicators[initial]*stable<indica_max)
			&& initial < steps-1){
			initial++;
			b_e=l_e[initial];
			b_N=l_N[initial];
			b_c=l_c[initial];
			b_i=initial;
		}
		//================result selection=====================//
		for(int j =initial+1; j<steps;j++){
			
			if ( _finite(l_e[j])&& l_e[j]>0 && _finite(l_N[j]) 
				&& indicators[initial]*stable>=indica_max 
				&& (pow(b_e,rate)/b_N >pow(l_e[j],rate)/l_N[j]||b_N<8)
			){
					if (second_check && b_N>=8)
					{
						float e, cardinal,best_thres=b_c.thres,candi_thres=l_c[j].thres;
						b_c.thres=candi_thres;
						mean_error(b_c,F1,F2,lists[j],e,cardinal);

						if (pow(abs(e),rate)/cardinal >pow(abs(l_e[j]),rate)/abs(l_N[j])){
							b_e=abs(l_e[j]);
							b_N=abs(l_N[j]);
							b_c=l_c[j];
							b_i=j;
						}else{
							b_e=e;
							b_N=cardinal;
						}
					}else
					{
						b_e=abs(l_e[j]);
						b_N=abs(l_N[j]);
						b_c=l_c[j];
						b_i=j;
					}
			} 
		}

}

