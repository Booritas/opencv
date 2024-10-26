/*
 * PnPProblem.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef PNPPROBLEM_H_
#define PNPPROBLEM_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Mesh.h"
#include "ModelRegistration.h"

class PnPProblem
{
public:
    explicit PnPProblem(const double param[]);  // custom constructor
    virtual ~PnPProblem();

    bool backproject2DPoint(const Mesh *mesh, const ncvslideio::Point2f &point2d, ncvslideio::Point3f &point3d);
    bool intersect_MollerTrumbore(Ray &R, Triangle &T, double *out);
    std::vector<ncvslideio::Point2f> verify_points(Mesh *mesh);
    ncvslideio::Point2f backproject3DPoint(const ncvslideio::Point3f &point3d);
    bool estimatePose(const std::vector<ncvslideio::Point3f> &list_points3d, const std::vector<ncvslideio::Point2f> &list_points2d, int flags);
    void estimatePoseRANSAC( const std::vector<ncvslideio::Point3f> &list_points3d, const std::vector<ncvslideio::Point2f> &list_points2d,
                             int flags, ncvslideio::Mat &inliers,
                             int iterationsCount, float reprojectionError, double confidence );

    ncvslideio::Mat get_A_matrix() const { return A_matrix_; }
    ncvslideio::Mat get_R_matrix() const { return R_matrix_; }
    ncvslideio::Mat get_t_matrix() const { return t_matrix_; }
    ncvslideio::Mat get_P_matrix() const { return P_matrix_; }

    void set_P_matrix( const ncvslideio::Mat &R_matrix, const ncvslideio::Mat &t_matrix);

private:
    /** The calibration matrix */
    ncvslideio::Mat A_matrix_;
    /** The computed rotation matrix */
    ncvslideio::Mat R_matrix_;
    /** The computed translation matrix */
    ncvslideio::Mat t_matrix_;
    /** The computed projection matrix */
    ncvslideio::Mat P_matrix_;
};

#endif /* PNPPROBLEM_H_ */
