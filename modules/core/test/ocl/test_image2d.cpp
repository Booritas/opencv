// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

TEST(Image2D, createAliasEmptyUMat)
{
    if (ncvslideio::ocl::haveOpenCL())
    {
        UMat um;
        EXPECT_FALSE(ncvslideio::ocl::Image2D::canCreateAlias(um));
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

TEST(Image2D, createImage2DWithEmptyUMat)
{
    if (ncvslideio::ocl::haveOpenCL())
    {
        UMat um;
        EXPECT_ANY_THROW(ncvslideio::ocl::Image2D image(um));
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

TEST(Image2D, createAlias)
{
    if (ncvslideio::ocl::haveOpenCL())
    {
        const ncvslideio::ocl::Device & d = ncvslideio::ocl::Device::getDefault();
        int minor = d.deviceVersionMinor(), major = d.deviceVersionMajor();

        // aliases is OpenCL 1.2 extension
        if (1 < major || (1 == major && 2 <= minor))
        {
            UMat um(128, 128, CV_8UC1);
            bool isFormatSupported = false, canCreateAlias = false;

            EXPECT_NO_THROW(isFormatSupported = ncvslideio::ocl::Image2D::isFormatSupported(CV_8U, 1, false));
            EXPECT_NO_THROW(canCreateAlias = ncvslideio::ocl::Image2D::canCreateAlias(um));

            if (isFormatSupported && canCreateAlias)
            {
                EXPECT_NO_THROW(ncvslideio::ocl::Image2D image(um, false, true));
            }
            else
                std::cout << "Impossible to create alias for selected image. Test skipped." << std::endl;
        }
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped" << std::endl;
}

TEST(Image2D, turnOffOpenCL)
{
    if (ncvslideio::ocl::haveOpenCL())
    {
        // save the current state
        bool useOCL = ncvslideio::ocl::useOpenCL();
        bool isFormatSupported = false;

        ncvslideio::ocl::setUseOpenCL(true);
        UMat um(128, 128, CV_8UC1);

        ncvslideio::ocl::setUseOpenCL(false);
        EXPECT_NO_THROW(isFormatSupported = ncvslideio::ocl::Image2D::isFormatSupported(CV_8U, 1, true));

        if (isFormatSupported)
        {
            EXPECT_NO_THROW(ncvslideio::ocl::Image2D image(um));
        }
        else
            std::cout << "CV_8UC1 is not supported for OpenCL images. Test skipped." << std::endl;

        // reset state to the previous one
        ncvslideio::ocl::setUseOpenCL(useOCL);
    }
    else
        std::cout << "OpenCL runtime not found. Test skipped." << std::endl;
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL