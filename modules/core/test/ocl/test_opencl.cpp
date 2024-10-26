// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../test_precomp.hpp"

#include <opencv2/core/ocl.hpp>

namespace opencv_test { namespace {

static void testOpenCLKernel(ncvslideio::ocl::Kernel& k)
{
    ASSERT_FALSE(k.empty());
    ncvslideio::UMat src(ncvslideio::Size(4096, 2048), CV_8UC1, ncvslideio::Scalar::all(100));
    ncvslideio::UMat dst(src.size(), CV_8UC1);
    size_t globalSize[2] = {(size_t)src.cols, (size_t)src.rows};
    size_t localSize[2] = {8, 8};
    int64 kernel_time = k.args(
            ncvslideio::ocl::KernelArg::ReadOnlyNoSize(src), // size is not used (similar to 'dst' size)
            ncvslideio::ocl::KernelArg::WriteOnly(dst),
            (int)5
        ).runProfiling(2, globalSize, localSize);
    ASSERT_GE(kernel_time, (int64)0);
    std::cout << "Kernel time: " << (kernel_time * 1e-6) << " ms" << std::endl;
    ncvslideio::Mat res, reference(src.size(), CV_8UC1, ncvslideio::Scalar::all(105));
    dst.copyTo(res);
    EXPECT_EQ(0, cvtest::norm(reference, res, ncvslideio::NORM_INF));
}

TEST(OpenCL, support_binary_programs)
{
    ncvslideio::ocl::Context ctx = ncvslideio::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    ncvslideio::ocl::Device device = ncvslideio::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        throw cvtest::SkipTestException("OpenCL compiler is not available");
    }
    std::vector<char> program_binary_code;

    ncvslideio::String module_name; // empty to disable OpenCL cache

    { // Generate program binary from OpenCL C source
        static const char* opencl_kernel_src =
"__kernel void test_kernel(__global const uchar* src, int src_step, int src_offset,\n"
"                          __global uchar* dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,\n"
"                          int c)\n"
"{\n"
"   int x = get_global_id(0);\n"
"   int y = get_global_id(1);\n"
"   if (x < dst_cols && y < dst_rows)\n"
"   {\n"
"       int src_idx = y * src_step + x + src_offset;\n"
"       int dst_idx = y * dst_step + x + dst_offset;\n"
"       dst[dst_idx] = src[src_idx] + c;\n"
"   }\n"
"}\n";
        ncvslideio::ocl::ProgramSource src(module_name, "simple", opencl_kernel_src, "");
        ncvslideio::String errmsg;
        ncvslideio::ocl::Program program(src, "", errmsg);
        ASSERT_TRUE(program.ptr() != NULL);
        ncvslideio::ocl::Kernel k("test_kernel", program);
        EXPECT_FALSE(k.empty());
        program.getBinary(program_binary_code);
        std::cout << "Program binary size: " << program_binary_code.size() << " bytes" << std::endl;
    }

    ncvslideio::ocl::Kernel k;

    { // Load program from binary (without sources)
        ASSERT_FALSE(program_binary_code.empty());
        ncvslideio::ocl::ProgramSource src = ncvslideio::ocl::ProgramSource::fromBinary(module_name, "simple_binary", (uchar*)&program_binary_code[0], program_binary_code.size(), "");
        ncvslideio::String errmsg;
        ncvslideio::ocl::Program program(src, "", errmsg);
        ASSERT_TRUE(program.ptr() != NULL);
        k.create("test_kernel", program);
    }

    testOpenCLKernel(k);
}


TEST(OpenCL, support_SPIR_programs)
{
    ncvslideio::ocl::Context ctx = ncvslideio::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    ncvslideio::ocl::Device device = ncvslideio::ocl::Device::getDefault();
    if (!device.isExtensionSupported("cl_khr_spir"))
    {
        throw cvtest::SkipTestException("'cl_khr_spir' extension is not supported by OpenCL device");
    }
    std::vector<char> program_binary_code;
    ncvslideio::String fname = ncvslideio::format("test_kernel.spir%d", device.addressBits());
    std::string full_path = cvtest::findDataFile(std::string("opencl/") + fname);

    {
        std::fstream f(full_path.c_str(), std::ios::in|std::ios::binary);
        ASSERT_TRUE(f.is_open());
        size_t pos = (size_t)f.tellg();
        f.seekg(0, std::fstream::end);
        size_t fileSize = (size_t)f.tellg();
        std::cout << "Program SPIR size: " << fileSize << " bytes" << std::endl;
        f.seekg(pos, std::fstream::beg);
        program_binary_code.resize(fileSize);
        f.read(&program_binary_code[0], fileSize);
        ASSERT_FALSE(f.fail());
    }

    ncvslideio::String module_name; // empty to disable OpenCL cache

    ncvslideio::ocl::Kernel k;

    { // Load program from SPIR format
        ASSERT_FALSE(program_binary_code.empty());
        ncvslideio::ocl::ProgramSource src = ncvslideio::ocl::ProgramSource::fromSPIR(module_name, "simple_spir", (uchar*)&program_binary_code[0], program_binary_code.size(), "");
        ncvslideio::String errmsg;
        ncvslideio::ocl::Program program(src, "", errmsg);
        if (program.ptr() == NULL && device.isAMD())
        {
            // https://community.amd.com/t5/opencl/spir-support-in-new-drivers-lost/td-p/170165
            throw cvtest::SkipTestException("Bypass AMD OpenCL runtime bug: 'cl_khr_spir' extension is declared, but it doesn't really work");
        }
        ASSERT_TRUE(program.ptr() != NULL);
        k.create("test_kernel", program);
    }

    testOpenCLKernel(k);
}


TEST(OpenCL, image2Dcount_regression_19334)
{
    ncvslideio::ocl::Context ctx = ncvslideio::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    ncvslideio::ocl::Device device = ncvslideio::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        throw cvtest::SkipTestException("OpenCL compiler is not available");
    }

    std::string module_name; // empty to disable OpenCL cache

    static const char* opencl_kernel_src =
"__kernel void test_kernel(int a,\n"
"                          __global const uchar* src0, int src0_step, int src0_offset, int src0_rows, int src0_cols,\n"
"                          __global const uchar* src1, int src1_step, int src1_offset, int src1_rows, int src1_cols,\n"
"                          __global const uchar* src2, int src2_step, int src2_offset, int src2_rows, int src2_cols,\n"
"                          __read_only image2d_t image)\n"
"{\n"
"}";
    ncvslideio::ocl::ProgramSource src(module_name, "test_opencl_image_arg", opencl_kernel_src, "");
    ncvslideio::String errmsg;
    ncvslideio::ocl::Program program(src, "", errmsg);
    ASSERT_TRUE(program.ptr() != NULL);
    ncvslideio::ocl::Kernel k("test_kernel", program);
    ASSERT_FALSE(k.empty());

    std::vector<UMat> images(4);
    for (size_t i = 0; i < images.size(); ++i)
        images[i] = UMat(10, 10, CV_8UC1);
    ncvslideio::ocl::Image2D image;
    try
    {
        ncvslideio::ocl::Image2D image_(images.back());
        image = image_;
    }
    catch (const ncvslideio::Exception&)
    {
        throw cvtest::SkipTestException("OpenCL images are not supported");
    }

    int nargs = 0;
    int a = 0;
    nargs = k.set(nargs, a);
    ASSERT_EQ(1, nargs);
    nargs = k.set(nargs, images[0]);
    ASSERT_EQ(6, nargs);
    nargs = k.set(nargs, images[1]);
    ASSERT_EQ(11, nargs);
    nargs = k.set(nargs, images[2]);
    ASSERT_EQ(16, nargs);

    // do not throw (issue of #19334)
    ASSERT_NO_THROW(nargs = k.set(nargs, image));
    ASSERT_EQ(17, nargs);

    // allow to replace image argument if kernel is not running
    UMat image2(10, 10, CV_8UC1);
    ASSERT_NO_THROW(nargs = k.set(16, ncvslideio::ocl::Image2D(image2)));
    ASSERT_EQ(17, nargs);
}


TEST(OpenCL, move_construct_assign)
{
    ncvslideio::ocl::Context ctx1 = ncvslideio::ocl::Context::getDefault();
    if (!ctx1.ptr())
    {
        throw cvtest::SkipTestException("OpenCL is not available");
    }
    void* const ctx_ptr = ctx1.ptr();
    ncvslideio::ocl::Context ctx2(std::move(ctx1));
    ASSERT_EQ(ctx1.ptr(), nullptr);
    ASSERT_EQ(ctx2.ptr(), ctx_ptr);
    ncvslideio::ocl::Context ctx3 = std::move(ctx2);
    ASSERT_EQ(ctx2.ptr(), nullptr);
    ASSERT_EQ(ctx3.ptr(), ctx_ptr);

    ncvslideio::ocl::Platform pl1 = ncvslideio::ocl::Platform::getDefault();
    void* const pl_ptr = pl1.ptr();
    ncvslideio::ocl::Platform pl2(std::move(pl1));
    ASSERT_EQ(pl1.ptr(), nullptr);
    ASSERT_EQ(pl2.ptr(), pl_ptr);
    ncvslideio::ocl::Platform pl3 = std::move(pl2);
    ASSERT_EQ(pl2.ptr(), nullptr);
    ASSERT_EQ(pl3.ptr(), pl_ptr);

    std::vector<ncvslideio::ocl::PlatformInfo> platformInfos;
    ncvslideio::ocl::getPlatfomsInfo(platformInfos);
    const ncvslideio::String pi_name = platformInfos[0].name();
    ncvslideio::ocl::PlatformInfo pinfo2(std::move(platformInfos[0]));
    ASSERT_EQ(platformInfos[0].name(), ncvslideio::String());
    ASSERT_EQ(pinfo2.name(), pi_name);
    ncvslideio::ocl::PlatformInfo pinfo3 = std::move(pinfo2);
    ASSERT_EQ(pinfo2.name(), ncvslideio::String());
    ASSERT_EQ(pinfo3.name(), pi_name);

    ncvslideio::ocl::Queue q1 = ncvslideio::ocl::Queue::getDefault();
    void* const q_ptr = q1.ptr();
    ncvslideio::ocl::Queue q2(std::move(q1));
    ASSERT_EQ(q1.ptr(), nullptr);
    ASSERT_EQ(q2.ptr(), q_ptr);
    ncvslideio::ocl::Queue q3 = std::move(q2);
    ASSERT_EQ(q2.ptr(), nullptr);
    ASSERT_EQ(q3.ptr(), q_ptr);

    ncvslideio::ocl::Device d1 = ncvslideio::ocl::Device::getDefault();
    if (!d1.compilerAvailable())
    {
        throw cvtest::SkipTestException("OpenCL compiler is not available");
    }
    void* const d_ptr = d1.ptr();
    ncvslideio::ocl::Device d2(std::move(d1));
    ASSERT_EQ(d1.ptr(), nullptr);
    ASSERT_EQ(d2.ptr(), d_ptr);
    ncvslideio::ocl::Device d3 = std::move(d2);
    ASSERT_EQ(d2.ptr(), nullptr);
    ASSERT_EQ(d3.ptr(), d_ptr);

    if (d3.imageSupport()) {
        ncvslideio::UMat umat1 = ncvslideio::UMat::ones(640, 480, CV_32FC1);
        ncvslideio::ocl::Image2D img1(umat1);
        void *const img_ptr = img1.ptr();
        ncvslideio::ocl::Image2D img2(std::move(img1));
        ASSERT_EQ(img1.ptr(), nullptr);
        ASSERT_EQ(img2.ptr(), img_ptr);
        ncvslideio::ocl::Image2D img3 = std::move(img2);
        ASSERT_EQ(img2.ptr(), nullptr);
        ASSERT_EQ(img3.ptr(), img_ptr);
    }

    static const char* opencl_kernel_src =
"__kernel void test_kernel(__global const uchar* src, int src_step, int src_offset,\n"
"                          __global uchar* dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,\n"
"                          int c)\n"
"{\n"
"   int x = get_global_id(0);\n"
"   int y = get_global_id(1);\n"
"   if (x < dst_cols && y < dst_rows)\n"
"   {\n"
"       int src_idx = y * src_step + x + src_offset;\n"
"       int dst_idx = y * dst_step + x + dst_offset;\n"
"       dst[dst_idx] = src[src_idx] + c;\n"
"   }\n"
"}\n";
    ncvslideio::String module_name; // empty to disable OpenCL cache
    ncvslideio::ocl::ProgramSource ps1(module_name, "move_construct_assign", opencl_kernel_src, "");
    ncvslideio::ocl::ProgramSource::Impl* const ps_ptr = ps1.getImpl();
    ncvslideio::ocl::ProgramSource ps2(std::move(ps1));
    ASSERT_EQ(ps1.getImpl(), nullptr);
    ASSERT_EQ(ps2.getImpl(), ps_ptr);
    ncvslideio::ocl::ProgramSource ps3 = std::move(ps2);
    ASSERT_EQ(ps2.getImpl(), nullptr);
    ASSERT_EQ(ps3.getImpl(), ps_ptr);

    ncvslideio::String errmsg;
    ncvslideio::ocl::Program prog1(ps3, "", errmsg);
    void* const prog_ptr = prog1.ptr();
    ASSERT_NE(prog_ptr, nullptr);
    ncvslideio::ocl::Program prog2(std::move(prog1));
    ASSERT_EQ(prog1.ptr(), nullptr);
    ASSERT_EQ(prog2.ptr(), prog_ptr);
    ncvslideio::ocl::Program prog3 = std::move(prog2);
    ASSERT_EQ(prog2.ptr(), nullptr);
    ASSERT_EQ(prog3.ptr(), prog_ptr);

    ncvslideio::ocl::Kernel k1("test_kernel", prog3);
    void* const k_ptr = k1.ptr();
    ASSERT_NE(k_ptr, nullptr);
    ncvslideio::ocl::Kernel k2(std::move(k1));
    ASSERT_EQ(k1.ptr(), nullptr);
    ASSERT_EQ(k2.ptr(), k_ptr);
    ncvslideio::ocl::Kernel k3 = std::move(k2);
    ASSERT_EQ(k2.ptr(), nullptr);
    ASSERT_EQ(k3.ptr(), k_ptr);

    testOpenCLKernel(k3);
}

}} // namespace
