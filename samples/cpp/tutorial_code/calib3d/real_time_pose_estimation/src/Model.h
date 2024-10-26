/*
 * Model.h
 *
 *  Created on: Apr 9, 2014
 *      Author: edgar
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class Model
{
public:
    Model();
    virtual ~Model();

    std::vector<ncvslideio::Point2f> get_points2d_in() const { return list_points2d_in_; }
    std::vector<ncvslideio::Point2f> get_points2d_out() const { return list_points2d_out_; }
    std::vector<ncvslideio::Point3f> get_points3d() const { return list_points3d_in_; }
    std::vector<ncvslideio::KeyPoint> get_keypoints() const { return list_keypoints_; }
    ncvslideio::Mat get_descriptors() const { return descriptors_; }
    int get_numDescriptors() const { return descriptors_.rows; }
    std::string get_trainingImagePath() const { return training_img_path_; }

    void add_correspondence(const ncvslideio::Point2f &point2d, const ncvslideio::Point3f &point3d);
    void add_outlier(const ncvslideio::Point2f &point2d);
    void add_descriptor(const ncvslideio::Mat &descriptor);
    void add_keypoint(const ncvslideio::KeyPoint &kp);
    void set_trainingImagePath(const std::string &path);

    void save(const std::string &path);
    void load(const std::string &path);

private:
    /** The current number of correspondences */
    int n_correspondences_;
    /** The list of 2D points on the model surface */
    std::vector<ncvslideio::KeyPoint> list_keypoints_;
    /** The list of 2D points on the model surface */
    std::vector<ncvslideio::Point2f> list_points2d_in_;
    /** The list of 2D points outside the model surface */
    std::vector<ncvslideio::Point2f> list_points2d_out_;
    /** The list of 3D points on the model surface */
    std::vector<ncvslideio::Point3f> list_points3d_in_;
    /** The list of 2D points descriptors */
    ncvslideio::Mat descriptors_;
    /** Path to the training image */
    std::string training_img_path_;
};

#endif /* OBJECTMODEL_H_ */
