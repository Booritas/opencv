/*
 * Utils.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>

#include <opencv2/features2d.hpp>
#include "PnPProblem.h"

// Draw a text with the question point
void drawQuestion(ncvslideio::Mat image, ncvslideio::Point3f point, ncvslideio::Scalar color);

// Draw a text with the number of entered points
void drawText(ncvslideio::Mat image, std::string text, ncvslideio::Scalar color);

// Draw a text with the number of entered points
void drawText2(ncvslideio::Mat image, std::string text, ncvslideio::Scalar color);

// Draw a text with the frame ratio
void drawFPS(ncvslideio::Mat image, double fps, ncvslideio::Scalar color);

// Draw a text with the frame ratio
void drawConfidence(ncvslideio::Mat image, double confidence, ncvslideio::Scalar color);

// Draw a text with the number of entered points
void drawCounter(ncvslideio::Mat image, int n, int n_max, ncvslideio::Scalar color);

// Draw the points and the coordinates
void drawPoints(ncvslideio::Mat image, std::vector<ncvslideio::Point2f> &list_points_2d, std::vector<ncvslideio::Point3f> &list_points_3d, ncvslideio::Scalar color);

// Draw only the 2D points
void draw2DPoints(ncvslideio::Mat image, std::vector<ncvslideio::Point2f> &list_points, ncvslideio::Scalar color);

// Draw an arrow into the image
void drawArrow(ncvslideio::Mat image, ncvslideio::Point2i p, ncvslideio::Point2i q, ncvslideio::Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);

// Draw the 3D coordinate axes
void draw3DCoordinateAxes(ncvslideio::Mat image, const std::vector<ncvslideio::Point2f> &list_points2d);

// Draw the object mesh
void drawObjectMesh(ncvslideio::Mat image, const Mesh *mesh, PnPProblem *pnpProblem, ncvslideio::Scalar color);

// Computes the norm of the translation error
double get_translation_error(const ncvslideio::Mat &t_true, const ncvslideio::Mat &t);

// Computes the norm of the rotation error
double get_rotation_error(const ncvslideio::Mat &R_true, const ncvslideio::Mat &R);

// Converts a given Rotation Matrix to Euler angles
ncvslideio::Mat rot2euler(const ncvslideio::Mat & rotationMatrix);

// Converts a given Euler angles to Rotation Matrix
ncvslideio::Mat euler2rot(const ncvslideio::Mat & euler);

// Converts a given string to an integer
int StringToInt ( const std::string &Text );

// Converts a given float to a string
std::string FloatToString ( float Number );

// Converts a given integer to a string
std::string IntToString ( int Number );

void createFeatures(const std::string &featureName, int numKeypoints, ncvslideio::Ptr<ncvslideio::Feature2D> &detector, ncvslideio::Ptr<ncvslideio::Feature2D> &descriptor);

ncvslideio::Ptr<ncvslideio::DescriptorMatcher> createMatcher(const std::string &featureName, bool useFLANN);

#endif /* UTILS_H_ */
