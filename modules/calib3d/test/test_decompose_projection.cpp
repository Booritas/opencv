/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_DecomposeProjectionMatrixTest : public cvtest::BaseTest
{
public:
    CV_DecomposeProjectionMatrixTest();
protected:
    void run(int);
};

CV_DecomposeProjectionMatrixTest::CV_DecomposeProjectionMatrixTest()
{
    test_case_count = 30;
}


void CV_DecomposeProjectionMatrixTest::run(int start_from)
{

    ts->set_failed_test_info(cvtest::TS::OK);

    ncvslideio::RNG& rng = ts->get_rng();
    int progress = 0;


    for (int iter = start_from; iter < test_case_count; ++iter)
    {
        ts->update_context(this, iter, true);
        progress = update_progress(progress, iter, test_case_count, 0);

        // Create the original (and random) camera matrix, rotation, and translation
        ncvslideio::Vec2d f, c;
        rng.fill(f, ncvslideio::RNG::UNIFORM, 300, 1000);
        rng.fill(c, ncvslideio::RNG::UNIFORM, 150, 600);

        double alpha = 0.01*rng.gaussian(1);

        ncvslideio::Matx33d origK(f(0), alpha*f(0), c(0),
                          0,          f(1), c(1),
                          0,             0,   1);


        ncvslideio::Vec3d rVec;
        rng.fill(rVec, ncvslideio::RNG::UNIFORM, -CV_PI, CV_PI);

        ncvslideio::Matx33d origR;
        ncvslideio::Rodrigues(rVec, origR); // TODO cvtest

        ncvslideio::Vec3d origT;
        rng.fill(origT, ncvslideio::RNG::NORMAL, 0, 1);


        // Compose the projection matrix
        ncvslideio::Matx34d P(3,4);
        hconcat(origK*origR, origK*origT, P);


        // Decompose
        ncvslideio::Matx33d K, R;
        ncvslideio::Vec4d homogCameraCenter;
        decomposeProjectionMatrix(P, K, R, homogCameraCenter);


        // Recover translation from the camera center
        ncvslideio::Vec3d cameraCenter(homogCameraCenter(0), homogCameraCenter(1), homogCameraCenter(2));
        cameraCenter /= homogCameraCenter(3);

        ncvslideio::Vec3d t = -R*cameraCenter;


        const double thresh = 1e-6;
        if (ncvslideio::norm(origK, K, ncvslideio::NORM_INF) > thresh)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            break;
        }

        if (ncvslideio::norm(origR, R, ncvslideio::NORM_INF) > thresh)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            break;
        }

        if (ncvslideio::norm(origT, t, ncvslideio::NORM_INF) > thresh)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            break;
        }

    }

}

TEST(Calib3d_DecomposeProjectionMatrix, accuracy)
{
    CV_DecomposeProjectionMatrixTest test;
    test.safe_run();
}

TEST(Calib3d_DecomposeProjectionMatrix, degenerate_cases)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            ncvslideio::Matx34d P;
            P(0, i) = 1;
            P(1, (i + j + 1) % 3) = 1;
            P(2, (i + 2 * j + 2) % 3) = 1;

            ncvslideio::Matx33d K, R;
            ncvslideio::Vec4d t;
            decomposeProjectionMatrix(P, K, R, t);
            EXPECT_LT(ncvslideio::norm(K * R, P.get_minor<3, 3>(0, 0), ncvslideio::NORM_INF), 1e-6);
        }
    }
}

TEST(Calib3d_DecomposeProjectionMatrix, bug_23733)
{
    ncvslideio::Matx34d P(52, -7, 4, 12,
                  -6, 49, 12, 8,
                  4, 17, 1, 0);
    P *= 1e-6;

    ncvslideio::Matx33d K, R;
    ncvslideio::Vec4d t;
    decomposeProjectionMatrix(P, K, R, t);

    EXPECT_LT(ncvslideio::norm(R.t() * R - ncvslideio::Matx33d::eye(), ncvslideio::NORM_INF), 1e-10);

    ncvslideio::Matx34d M;
    ncvslideio::hconcat(R, -R * ncvslideio::Vec3d(t[0] / t[3], t[1] / t[3], t[2] / t[3]), M);

    ncvslideio::Matx34d P_recompose = K * M;
    EXPECT_LT(ncvslideio::norm(P_recompose - P, ncvslideio::NORM_INF), 1e-16);
}

}} // namespace
