#include <iostream>

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, const char* argv[])
{
    if (argc != 2)
        return -1;

    const std::string fname(argv[1]);

    ncvslideio::namedWindow("CPU", ncvslideio::WINDOW_NORMAL);
#if defined(HAVE_OPENGL)
    ncvslideio::namedWindow("GPU", ncvslideio::WINDOW_OPENGL);
    ncvslideio::cuda::setGlDevice();
#else
    ncvslideio::namedWindow("GPU", ncvslideio::WINDOW_NORMAL);
#endif

    ncvslideio::TickMeter tm;
    ncvslideio::Mat frame;
    ncvslideio::VideoCapture reader(fname);
    for (;;)
    {
        if (!reader.read(frame))
            break;
        ncvslideio::imshow("CPU", frame);
        if (ncvslideio::waitKey(3) > 0)
            break;
    }

    ncvslideio::cuda::GpuMat d_frame;
    ncvslideio::Ptr<ncvslideio::cudacodec::VideoReader> d_reader = ncvslideio::cudacodec::createVideoReader(fname);
    for (;;)
    {
        if (!d_reader->nextFrame(d_frame))
            break;
#if defined(HAVE_OPENGL)
        ncvslideio::imshow("GPU", ncvslideio::ogl::Texture2D(d_frame));
#else
        d_frame.download(frame);
        ncvslideio::imshow("GPU", frame);
#endif
        if (ncvslideio::waitKey(3) > 0)
            break;
    }

    return 0;
}

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
