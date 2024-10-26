#include "perf_precomp.hpp"
#include "../test/common/gapi_tests_common.hpp"

namespace opencv_test
{

struct SobelEdgeDetector:  public TestPerfParams<ncvslideio::Size> {};
PERF_TEST_P_(SobelEdgeDetector, Fluid)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    GMat in;
    GMat gx  = gapi::Sobel(in, CV_32F, 1, 0);
    GMat gy  = gapi::Sobel(in, CV_32F, 0, 1);
    GMat mag = gapi::sqrt(gapi::mul(gx, gx) + gapi::mul(gy, gy));
    GMat out = gapi::convertTo(mag, CV_8U);
    GComputation sobel(in, out);
    auto pkg = gapi::combine(gapi::core::fluid::kernels(),
                             gapi::imgproc::fluid::kernels());
    auto cc = sobel.compile(ncvslideio::descr_of(in_mat1),
                            ncvslideio::compile_args(ncvslideio::gapi::use_only{pkg}));
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P_(SobelEdgeDetector, OpenCV)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    Mat gx, gy;
    Mat mag;
    auto cc = [&](const ncvslideio::Mat &in_mat, ncvslideio::Mat &out_mat) {
        using namespace ncvslideio;

        Sobel(in_mat, gx, CV_32F, 1, 0);
        Sobel(in_mat, gy, CV_32F, 0, 1);
        sqrt(gx.mul(gx) + gy.mul(gy), mag);
        mag.convertTo(out_mat, CV_8U);
    };
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P_(SobelEdgeDetector, OpenCV_Smarter)
{
    Size sz = GetParam();
    initMatsRandU(CV_8UC3, sz, CV_8UC3, false);

    Mat gx, gy;
    Mat ggx, ggy;
    Mat sum;
    Mat mag;

    auto cc = [&](const ncvslideio::Mat &in_mat, ncvslideio::Mat &out_mat) {
        ncvslideio::Sobel(in_mat, gx, CV_32F, 1, 0);
        ncvslideio::Sobel(in_mat, gy, CV_32F, 0, 1);
        ncvslideio::multiply(gx, gx, ggx);
        ncvslideio::multiply(gy, gy, ggy);
        ncvslideio::add(ggx, ggy, sum);
        ncvslideio::sqrt(sum, mag);
        mag.convertTo(out_mat, CV_8U);
    };
    cc(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        cc(in_mat1, out_mat_gapi);
    }
    SANITY_CHECK_NOTHING();
}
INSTANTIATE_TEST_CASE_P(Benchmark, SobelEdgeDetector,
                        Values(ncvslideio::Size(320, 240),
                               ncvslideio::Size(640, 480),
                               ncvslideio::Size(1280, 720),
                               ncvslideio::Size(1920, 1080),
                               ncvslideio::Size(3840, 2170)));

} // opencv_test
