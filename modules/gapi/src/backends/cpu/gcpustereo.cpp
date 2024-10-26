// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/stereo.hpp>
#include <opencv2/gapi/cpu/stereo.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#ifdef HAVE_OPENCV_CALIB3D
#include <opencv2/calib3d.hpp>
#endif // HAVE_OPENCV_CALIB3D

#ifdef HAVE_OPENCV_CALIB3D

/** @brief Structure for the Stereo operation setup parameters.*/
struct GAPI_EXPORTS StereoSetup {
    double baseline;
    double focus;
    ncvslideio::Ptr<ncvslideio::StereoBM> stereoBM;
};

namespace {
ncvslideio::Mat calcDepth(const ncvslideio::Mat &left, const ncvslideio::Mat &right,
                  const StereoSetup &ss) {
    constexpr int DISPARITY_SHIFT_16S = 4;
    ncvslideio::Mat disp;
    ss.stereoBM->compute(left, right, disp);
    disp.convertTo(disp, CV_32FC1, 1./(1 << DISPARITY_SHIFT_16S), 0);
    return (ss.focus * ss.baseline) / disp;
}
} // anonymous namespace

GAPI_OCV_KERNEL_ST(GCPUStereo, ncvslideio::gapi::calib3d::GStereo, StereoSetup)
{
    static void setup(const ncvslideio::GMatDesc&, const ncvslideio::GMatDesc&,
                      const ncvslideio::gapi::StereoOutputFormat,
                      std::shared_ptr<StereoSetup> &stereoSetup,
                      const ncvslideio::GCompileArgs &compileArgs) {
        auto stereoInit = ncvslideio::gapi::getCompileArg<ncvslideio::gapi::calib3d::cpu::StereoInitParam>(compileArgs)
            .value_or(ncvslideio::gapi::calib3d::cpu::StereoInitParam{});

        StereoSetup ss{stereoInit.baseline,
                       stereoInit.focus,
                       ncvslideio::StereoBM::create(stereoInit.numDisparities,
                       stereoInit.blockSize)};
        stereoSetup = std::make_shared<StereoSetup>(ss);
    }
    static void run(const ncvslideio::Mat& left,
                    const ncvslideio::Mat& right,
                    const ncvslideio::gapi::StereoOutputFormat oF,
                    ncvslideio::Mat& out_mat,
                    const StereoSetup &stereoSetup) {
        switch(oF){
            case ncvslideio::gapi::StereoOutputFormat::DEPTH_FLOAT16:
                calcDepth(left, right, stereoSetup).convertTo(out_mat, CV_16FC1);
                break;
            case ncvslideio::gapi::StereoOutputFormat::DEPTH_FLOAT32:
                calcDepth(left, right, stereoSetup).copyTo(out_mat);
                break;
            case ncvslideio::gapi::StereoOutputFormat::DISPARITY_FIXED16_12_4:
                stereoSetup.stereoBM->compute(left, right, out_mat);
                break;
            case ncvslideio::gapi::StereoOutputFormat::DISPARITY_FIXED16_11_5:
                GAPI_Error("This case may be supported in future.");
            default:
                GAPI_Error("Unknown output format!");
        }
    }
};

ncvslideio::GKernelPackage ncvslideio::gapi::calib3d::cpu::kernels() {
    static auto pkg = ncvslideio::gapi::kernels<GCPUStereo>();
    return pkg;
}

#else

ncvslideio::GKernelPackage ncvslideio::gapi::calib3d::cpu::kernels()
{
    return GKernelPackage();
}

#endif // HAVE_OPENCV_CALIB3D
