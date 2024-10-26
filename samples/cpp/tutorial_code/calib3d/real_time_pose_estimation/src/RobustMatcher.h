/*
 * RobustMatcher.h
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#ifndef ROBUSTMATCHER_H_
#define ROBUSTMATCHER_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class RobustMatcher {
public:
    RobustMatcher() : detector_(), extractor_(), matcher_(),
        ratio_(0.8f), training_img_(), img_matching_()
    {
        // ORB is the default feature
        detector_ = ncvslideio::ORB::create();
        extractor_ = ncvslideio::ORB::create();

        // BruteFroce matcher with Norm Hamming is the default matcher
        matcher_ = ncvslideio::makePtr<ncvslideio::BFMatcher>((int)ncvslideio::NORM_HAMMING, false);

    }
    virtual ~RobustMatcher();

    // Set the feature detector
    void setFeatureDetector(const ncvslideio::Ptr<ncvslideio::FeatureDetector>& detect) {  detector_ = detect; }

    // Set the descriptor extractor
    void setDescriptorExtractor(const ncvslideio::Ptr<ncvslideio::DescriptorExtractor>& desc) { extractor_ = desc; }

    // Set the matcher
    void setDescriptorMatcher(const ncvslideio::Ptr<ncvslideio::DescriptorMatcher>& match) {  matcher_ = match; }

    // Compute the keypoints of an image
    void computeKeyPoints( const ncvslideio::Mat& image, std::vector<ncvslideio::KeyPoint>& keypoints);

    // Compute the descriptors of an image given its keypoints
    void computeDescriptors( const ncvslideio::Mat& image, std::vector<ncvslideio::KeyPoint>& keypoints, ncvslideio::Mat& descriptors);

    ncvslideio::Mat getImageMatching() const { return img_matching_; }

    // Set ratio parameter for the ratio test
    void setRatio( float rat) { ratio_ = rat; }

    void setTrainingImage(const ncvslideio::Mat &img) { training_img_ = img; }

    // Clear matches for which NN ratio is > than threshold
    // return the number of removed points
    // (corresponding entries being cleared,
    // i.e. size will be 0)
    int ratioTest(std::vector<std::vector<ncvslideio::DMatch> > &matches);

    // Insert symmetrical matches in symMatches vector
    void symmetryTest( const std::vector<std::vector<ncvslideio::DMatch> >& matches1,
                       const std::vector<std::vector<ncvslideio::DMatch> >& matches2,
                       std::vector<ncvslideio::DMatch>& symMatches );

    // Match feature points using ratio and symmetry test
    void robustMatch( const ncvslideio::Mat& frame, std::vector<ncvslideio::DMatch>& good_matches,
                      std::vector<ncvslideio::KeyPoint>& keypoints_frame,
                      const ncvslideio::Mat& descriptors_model,
                      const std::vector<ncvslideio::KeyPoint>& keypoints_model);

    // Match feature points using ratio test
    void fastRobustMatch( const ncvslideio::Mat& frame, std::vector<ncvslideio::DMatch>& good_matches,
                          std::vector<ncvslideio::KeyPoint>& keypoints_frame,
                          const ncvslideio::Mat& descriptors_model,
                          const std::vector<ncvslideio::KeyPoint>& keypoints_model);

private:
    // pointer to the feature point detector object
    ncvslideio::Ptr<ncvslideio::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    ncvslideio::Ptr<ncvslideio::DescriptorExtractor> extractor_;
    // pointer to the matcher object
    ncvslideio::Ptr<ncvslideio::DescriptorMatcher> matcher_;
    // max ratio between 1st and 2nd NN
    float ratio_;
    // training image
    ncvslideio::Mat training_img_;
    // matching image
    ncvslideio::Mat img_matching_;
};

#endif /* ROBUSTMATCHER_H_ */
