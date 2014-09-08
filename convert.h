﻿/** @Converting functions between structures of openCV and KVLD suitable structures
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2007-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/
#include <vector>
#include <cv.hpp>

#include "demo/libImage/image.hpp"
#include "extras/sift/demo_lib_sift.h"
#include "./kvld/LWImage.h"
#include "./demo/libImage/image.hpp"
#include "./demo/libImage/image_io.hpp"

#include "./kvld/algorithm.h"


//typedef std::pair<size_t,size_t> Pair;


int Convert_image(const cv::Mat& In, Image<float> & imag);//convert only gray scale image of opencv
LWImage<float> Convert_image(const cv::Mat& In);
cv::Mat Convert_image(const LWImage<float> & image,int spline);

int Convert_detectors(const  std::vector<cv::KeyPoint>& feat1,std::vector<keypoint>& F1);//convert openCV detectors to KVLD suitable detectors
int Convert_features(const  std::vector<cv::KeyPoint>& feat1,std::vector<keypoint>& F1, cv::Mat descriptors1);
int Convert_matches(const std::vector<cv::DMatch>& matches, std::vector<Pair>& matchesPair);

cv::Mat convert(libNumerics::matrix<double> M);
libNumerics::matrix<double> convert(cv::Mat M);

int read_detectors(const std::string& filename ,  std::vector<cv::KeyPoint>& feat);
int read_matches(const std::string& filename , std::vector<cv::DMatch>& matches);
