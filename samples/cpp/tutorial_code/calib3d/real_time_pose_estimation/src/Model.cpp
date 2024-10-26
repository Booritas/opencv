/*
 * Model.cpp
 *
 *  Created on: Apr 9, 2014
 *      Author: edgar
 */

#include "Model.h"
#include "CsvWriter.h"

Model::Model() : n_correspondences_(0), list_points2d_in_(0), list_points2d_out_(0), list_points3d_in_(0), training_img_path_()
{
}

Model::~Model()
{
    // TODO Auto-generated destructor stub
}

void Model::add_correspondence(const ncvslideio::Point2f &point2d, const ncvslideio::Point3f &point3d)
{
    list_points2d_in_.push_back(point2d);
    list_points3d_in_.push_back(point3d);
    n_correspondences_++;
}

void Model::add_outlier(const ncvslideio::Point2f &point2d)
{
    list_points2d_out_.push_back(point2d);
}

void Model::add_descriptor(const ncvslideio::Mat &descriptor)
{
    descriptors_.push_back(descriptor);
}

void Model::add_keypoint(const ncvslideio::KeyPoint &kp)
{
    list_keypoints_.push_back(kp);
}

void Model::set_trainingImagePath(const std::string &path)
{
    training_img_path_ = path;
}

/** Save a YAML file and fill the object mesh */
void Model::save(const std::string &path)
{
    ncvslideio::Mat points3dmatrix = ncvslideio::Mat(list_points3d_in_);
    ncvslideio::Mat points2dmatrix = ncvslideio::Mat(list_points2d_in_);

    ncvslideio::FileStorage storage(path, ncvslideio::FileStorage::WRITE);
    storage << "points_3d" << points3dmatrix;
    storage << "points_2d" << points2dmatrix;
    storage << "keypoints" << list_keypoints_;
    storage << "descriptors" << descriptors_;
    storage << "training_image_path" << training_img_path_;

    storage.release();
}

/** Load a YAML file using OpenCv functions **/
void Model::load(const std::string &path)
{
    ncvslideio::Mat points3d_mat;

    ncvslideio::FileStorage storage(path, ncvslideio::FileStorage::READ);
    storage["points_3d"] >> points3d_mat;
    storage["descriptors"] >> descriptors_;
    if (!storage["keypoints"].empty())
    {
        storage["keypoints"] >> list_keypoints_;
    }
    if (!storage["training_image_path"].empty())
    {
        storage["training_image_path"] >> training_img_path_;
    }

    points3d_mat.copyTo(list_points3d_in_);

    storage.release();
}
