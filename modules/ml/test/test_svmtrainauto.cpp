// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using ncvslideio::ml::SVM;
using ncvslideio::ml::TrainData;

static Ptr<TrainData> makeRandomData(int datasize)
{
    ncvslideio::Mat samples = ncvslideio::Mat::zeros( datasize, 2, CV_32FC1 );
    ncvslideio::Mat responses = ncvslideio::Mat::zeros( datasize, 1, CV_32S );
    RNG &rng = ncvslideio::theRNG();
    for (int i = 0; i < datasize; ++i)
    {
        int response = rng.uniform(0, 2);  // Random from {0, 1}.
        samples.at<float>( i, 0 ) = rng.uniform(0.f, 0.5f) + response * 0.5f;
        samples.at<float>( i, 1 ) = rng.uniform(0.f, 0.5f) + response * 0.5f;
        responses.at<int>( i, 0 ) = response;
    }
    return TrainData::create( samples, ncvslideio::ml::ROW_SAMPLE, responses );
}

static Ptr<TrainData> makeCircleData(int datasize, float scale_factor, float radius)
{
    // Populate samples with data that can be split into two concentric circles
    ncvslideio::Mat samples = ncvslideio::Mat::zeros( datasize, 2, CV_32FC1 );
    ncvslideio::Mat responses = ncvslideio::Mat::zeros( datasize, 1, CV_32S );
    for (int i = 0; i < datasize; i+=2)
    {
        const float pi = 3.14159f;
        const float angle_rads = (i/datasize) * pi;
        const float x = radius * cos(angle_rads);
        const float y = radius * cos(angle_rads);

        // Larger circle
        samples.at<float>( i, 0 ) = x;
        samples.at<float>( i, 1 ) = y;
        responses.at<int>( i, 0 ) = 0;

        // Smaller circle
        samples.at<float>( i + 1, 0 ) = x * scale_factor;
        samples.at<float>( i + 1, 1 ) = y * scale_factor;
        responses.at<int>( i + 1, 0 ) = 1;
    }
    return TrainData::create( samples, ncvslideio::ml::ROW_SAMPLE, responses );
}

static Ptr<TrainData> makeRandomData2(int datasize)
{
    ncvslideio::Mat samples = ncvslideio::Mat::zeros( datasize, 2, CV_32FC1 );
    ncvslideio::Mat responses = ncvslideio::Mat::zeros( datasize, 1, CV_32S );
    RNG &rng = ncvslideio::theRNG();
    for (int i = 0; i < datasize; ++i)
    {
        int response = rng.uniform(0, 2);  // Random from {0, 1}.
        samples.at<float>( i, 0 ) = 0;
        samples.at<float>( i, 1 ) = (0.5f - response) * rng.uniform(0.f, 1.2f) + response;
        responses.at<int>( i, 0 ) = response;
    }
    return TrainData::create( samples, ncvslideio::ml::ROW_SAMPLE, responses );
}

//==================================================================================================

TEST(ML_SVM, trainauto)
{
    const int datasize = 100;
    ncvslideio::Ptr<TrainData> data = makeRandomData(datasize);
    ASSERT_TRUE(data);
    ncvslideio::Ptr<SVM> svm = SVM::create();
    ASSERT_TRUE(svm);
    svm->trainAuto( data, 10 );  // 2-fold cross validation.

    float test_data0[2] = {0.25f, 0.25f};
    ncvslideio::Mat test_point0 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data0 );
    float result0 = svm->predict( test_point0 );
    float test_data1[2] = {0.75f, 0.75f};
    ncvslideio::Mat test_point1 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data1 );
    float result1 = svm->predict( test_point1 );

    EXPECT_NEAR(result0, 0, 0.001);
    EXPECT_NEAR(result1, 1, 0.001);
}

TEST(ML_SVM, trainauto_sigmoid)
{
    const int datasize = 100;
    const float scale_factor = 0.5;
    const float radius = 2.0;
    ncvslideio::Ptr<TrainData> data = makeCircleData(datasize, scale_factor, radius);
    ASSERT_TRUE(data);

    ncvslideio::Ptr<SVM> svm = SVM::create();
    ASSERT_TRUE(svm);
    svm->setKernel(SVM::SIGMOID);
    svm->setGamma(10.0);
    svm->setCoef0(-10.0);
    svm->trainAuto( data, 10 );  // 2-fold cross validation.

    float test_data0[2] = {radius, radius};
    ncvslideio::Mat test_point0 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data0 );
    EXPECT_FLOAT_EQ(svm->predict( test_point0 ), 0);

    float test_data1[2] = {scale_factor * radius, scale_factor * radius};
    ncvslideio::Mat test_point1 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data1 );
    EXPECT_FLOAT_EQ(svm->predict( test_point1 ), 1);
}

TEST(ML_SVM, trainAuto_regression_5369)
{
    const int datasize = 100;
    Ptr<TrainData> data = makeRandomData2(datasize);
    ncvslideio::Ptr<SVM> svm = SVM::create();
    svm->trainAuto( data, 10 );  // 2-fold cross validation.

    float test_data0[2] = {0.25f, 0.25f};
    ncvslideio::Mat test_point0 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data0 );
    float result0 = svm->predict( test_point0 );
    float test_data1[2] = {0.75f, 0.75f};
    ncvslideio::Mat test_point1 = ncvslideio::Mat( 1, 2, CV_32FC1, test_data1 );
    float result1 = svm->predict( test_point1 );

    EXPECT_EQ(0., result0);
    EXPECT_EQ(1., result1);
}

TEST(ML_SVM, getSupportVectors)
{
    // Set up training data
    int labels[4] = {1, -1, -1, -1};
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);

    Ptr<SVM> svm = SVM::create();
    ASSERT_TRUE(svm);
    svm->setType(SVM::C_SVC);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // Test retrieval of SVs and compressed SVs on linear SVM
    svm->setKernel(SVM::LINEAR);
    svm->train(trainingDataMat, ncvslideio::ml::ROW_SAMPLE, labelsMat);

    Mat sv = svm->getSupportVectors();
    EXPECT_EQ(1, sv.rows);    // by default compressed SV returned
    sv = svm->getUncompressedSupportVectors();
    EXPECT_EQ(3, sv.rows);

    // Test retrieval of SVs and compressed SVs on non-linear SVM
    svm->setKernel(SVM::POLY);
    svm->setDegree(2);
    svm->train(trainingDataMat, ncvslideio::ml::ROW_SAMPLE, labelsMat);

    sv = svm->getSupportVectors();
    EXPECT_EQ(3, sv.rows);
    sv = svm->getUncompressedSupportVectors();
    EXPECT_EQ(0, sv.rows);    // inapplicable for non-linear SVMs
}

}} // namespace
