// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

void Mat_to_vector_int(ncvslideio::Mat& mat, std::vector<int>& v_int);
void vector_int_to_Mat(std::vector<int>& v_int, ncvslideio::Mat& mat);

void Mat_to_vector_double(ncvslideio::Mat& mat, std::vector<double>& v_double);
void vector_double_to_Mat(std::vector<double>& v_double, ncvslideio::Mat& mat);

void Mat_to_vector_float(ncvslideio::Mat& mat, std::vector<float>& v_float);
void vector_float_to_Mat(std::vector<float>& v_float, ncvslideio::Mat& mat);

void Mat_to_vector_uchar(ncvslideio::Mat& mat, std::vector<uchar>& v_uchar);
void vector_uchar_to_Mat(std::vector<uchar>& v_uchar, ncvslideio::Mat& mat);

void Mat_to_vector_char(ncvslideio::Mat& mat, std::vector<char>& v_char);
void vector_char_to_Mat(std::vector<char>& v_char, ncvslideio::Mat& mat);

void Mat_to_vector_Rect(ncvslideio::Mat& mat, std::vector<ncvslideio::Rect>& v_rect);
void vector_Rect_to_Mat(std::vector<ncvslideio::Rect>& v_rect, ncvslideio::Mat& mat);

void Mat_to_vector_Rect2d(ncvslideio::Mat& mat, std::vector<ncvslideio::Rect2d>& v_rect);
void vector_Rect2d_to_Mat(std::vector<ncvslideio::Rect2d>& v_rect, ncvslideio::Mat& mat);

void Mat_to_vector_RotatedRect(ncvslideio::Mat& mat, std::vector<ncvslideio::RotatedRect>& v_rect);
void vector_RotatedRect_to_Mat(std::vector<ncvslideio::RotatedRect>& v_rect, ncvslideio::Mat& mat);

void Mat_to_vector_Point(ncvslideio::Mat& mat, std::vector<ncvslideio::Point>& v_point);
void Mat_to_vector_Point2f(ncvslideio::Mat& mat, std::vector<ncvslideio::Point2f>& v_point);
void Mat_to_vector_Point2d(ncvslideio::Mat& mat, std::vector<ncvslideio::Point2d>& v_point);
void Mat_to_vector_Point3i(ncvslideio::Mat& mat, std::vector<ncvslideio::Point3i>& v_point);
void Mat_to_vector_Point3f(ncvslideio::Mat& mat, std::vector<ncvslideio::Point3f>& v_point);
void Mat_to_vector_Point3d(ncvslideio::Mat& mat, std::vector<ncvslideio::Point3d>& v_point);

void vector_Point_to_Mat(std::vector<ncvslideio::Point>& v_point, ncvslideio::Mat& mat);
void vector_Point2f_to_Mat(std::vector<ncvslideio::Point2f>& v_point, ncvslideio::Mat& mat);
void vector_Point2d_to_Mat(std::vector<ncvslideio::Point2d>& v_point, ncvslideio::Mat& mat);
void vector_Point3i_to_Mat(std::vector<ncvslideio::Point3i>& v_point, ncvslideio::Mat& mat);
void vector_Point3f_to_Mat(std::vector<ncvslideio::Point3f>& v_point, ncvslideio::Mat& mat);
void vector_Point3d_to_Mat(std::vector<ncvslideio::Point3d>& v_point, ncvslideio::Mat& mat);

void vector_Vec4i_to_Mat(std::vector<ncvslideio::Vec4i>& v_vec, ncvslideio::Mat& mat);
void vector_Vec4f_to_Mat(std::vector<ncvslideio::Vec4f>& v_vec, ncvslideio::Mat& mat);
void vector_Vec6f_to_Mat(std::vector<ncvslideio::Vec6f>& v_vec, ncvslideio::Mat& mat);

void Mat_to_vector_Mat(ncvslideio::Mat& mat, std::vector<ncvslideio::Mat>& v_mat);
void vector_Mat_to_Mat(std::vector<ncvslideio::Mat>& v_mat, ncvslideio::Mat& mat);

void Mat_to_vector_vector_char(ncvslideio::Mat& mat, std::vector< std::vector< char > >& vv_ch);
void vector_vector_char_to_Mat(std::vector< std::vector< char > >& vv_ch, ncvslideio::Mat& mat);

void Mat_to_vector_vector_Point(ncvslideio::Mat& mat, std::vector< std::vector< ncvslideio::Point > >& vv_pt);
void vector_vector_Point_to_Mat(std::vector< std::vector< ncvslideio::Point > >& vv_pt, ncvslideio::Mat& mat);

void Mat_to_vector_vector_Point2f(ncvslideio::Mat& mat, std::vector< std::vector< ncvslideio::Point2f > >& vv_pt);
void vector_vector_Point2f_to_Mat(std::vector< std::vector< ncvslideio::Point2f > >& vv_pt, ncvslideio::Mat& mat);

void Mat_to_vector_vector_Point3f(ncvslideio::Mat& mat, std::vector< std::vector< ncvslideio::Point3f > >& vv_pt);
void vector_vector_Point3f_to_Mat(std::vector< std::vector< ncvslideio::Point3f > >& vv_pt, ncvslideio::Mat& mat);
