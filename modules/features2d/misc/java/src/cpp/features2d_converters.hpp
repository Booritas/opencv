#ifndef __FEATURES2D_CONVERTERS_HPP__
#define __FEATURES2D_CONVERTERS_HPP__

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

void Mat_to_vector_KeyPoint(ncvslideio::Mat& mat, std::vector<ncvslideio::KeyPoint>& v_kp);
void vector_KeyPoint_to_Mat(std::vector<ncvslideio::KeyPoint>& v_kp, ncvslideio::Mat& mat);

void Mat_to_vector_DMatch(ncvslideio::Mat& mat, std::vector<ncvslideio::DMatch>& v_dm);
void vector_DMatch_to_Mat(std::vector<ncvslideio::DMatch>& v_dm, ncvslideio::Mat& mat);

void Mat_to_vector_vector_KeyPoint(ncvslideio::Mat& mat, std::vector< std::vector< ncvslideio::KeyPoint > >& vv_kp);
void vector_vector_KeyPoint_to_Mat(std::vector< std::vector< ncvslideio::KeyPoint > >& vv_kp, ncvslideio::Mat& mat);

void Mat_to_vector_vector_DMatch(ncvslideio::Mat& mat, std::vector< std::vector< ncvslideio::DMatch > >& vv_dm);
void vector_vector_DMatch_to_Mat(std::vector< std::vector< ncvslideio::DMatch > >& vv_dm, ncvslideio::Mat& mat);


#endif
