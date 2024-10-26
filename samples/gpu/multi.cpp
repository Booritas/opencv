/* This sample demonstrates the way you can perform independent tasks
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/cudaarithm.hpp"

#if !defined(HAVE_CUDA)

int main()
{
    std::cout << "CUDA support is required (OpenCV CMake parameter 'WITH_CUDA' must be true)." << std::endl;
    return 0;
}

#else

using namespace std;
using namespace ncvslideio;
using namespace ncvslideio::cuda;

struct Worker : public ncvslideio::ParallelLoopBody
{
    void operator()(const Range& r) const CV_OVERRIDE
    {
        for (int i = r.start; i < r.end; ++i) { this->operator()(i); }
    }
    void operator()(int device_id) const;
};

int main()
{
    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices < 2)
    {
        std::cout << "Two or more GPUs are required\n";
        return -1;
    }
    for (int i = 0; i < num_devices; ++i)
    {
        ncvslideio::cuda::printShortCudaDeviceInfo(i);

        DeviceInfo dev_info(i);
        if (!dev_info.isCompatible())
        {
            std::cout << "CUDA module isn't built for GPU #" << i << " ("
                 << dev_info.name() << ", CC " << dev_info.majorVersion()
                 << dev_info.minorVersion() << "\n";
            return -1;
        }
    }

    // Execute calculation in two threads using two GPUs
    ncvslideio::Range devices(0, 2);
    ncvslideio::parallel_for_(devices, Worker(), devices.size());

    return 0;
}


void Worker::operator()(int device_id) const
{
    setDevice(device_id);

    Mat src(1000, 1000, CV_32F);
    Mat dst;

    RNG rng(0);
    rng.fill(src, RNG::UNIFORM, 0, 1);

    // CPU works
    ncvslideio::transpose(src, dst);

    // GPU works
    GpuMat d_src(src);
    GpuMat d_dst;
    cuda::transpose(d_src, d_dst);

    // Check results
    bool passed = ncvslideio::norm(dst - Mat(d_dst), NORM_INF) < 1e-3;
    std::cout << "GPU #" << device_id << " (" << DeviceInfo().name() << "): "
        << (passed ? "passed" : "FAILED") << endl;

    // Deallocate data here, otherwise deallocation will be performed
    // after context is extracted from the stack
    d_src.release();
    d_dst.release();
}

#endif
