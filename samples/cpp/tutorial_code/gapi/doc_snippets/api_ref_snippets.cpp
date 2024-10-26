#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>

static void gscalar_example()
{
    //! [gscalar_implicit]
    ncvslideio::GMat a;
    ncvslideio::GMat b = a + 1;
    //! [gscalar_implicit]
}

static void typed_example()
{
    const ncvslideio::Size sz(32, 32);
    ncvslideio::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    //! [Untyped_Example]
    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cvtU([]()
    {
        ncvslideio::GMat in1, in2;
        ncvslideio::GMat out = ncvslideio::gapi::add(in1, in2);
        return ncvslideio::GComputation({in1, in2}, {out});
    });
    std::vector<ncvslideio::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<ncvslideio::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);
    //! [Untyped_Example]

    //! [Typed_Example]
    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<ncvslideio::GMat (ncvslideio::GMat, ncvslideio::GMat)> cvtT([](ncvslideio::GMat m1, ncvslideio::GMat m2)
    {
        return m1+m2;
    });
    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);

    auto cvtTC =  cvtT.compile(ncvslideio::descr_of(in_mat1), ncvslideio::descr_of(in_mat2));
    cvtTC(in_mat1, in_mat2, out_mat_typed2);
    //! [Typed_Example]
}

static void bind_serialization_example()
{
    // ! [bind after deserialization]
    ncvslideio::GCompiled compd;
    std::vector<char> bytes;
    auto graph = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(bytes);
    auto meta = ncvslideio::gapi::deserialize<ncvslideio::GMetaArgs>(bytes);

    compd = graph.compile(std::move(meta), ncvslideio::compile_args());
    auto in_args  = ncvslideio::gapi::deserialize<ncvslideio::GRunArgs>(bytes);
    auto out_args = ncvslideio::gapi::deserialize<ncvslideio::GRunArgs>(bytes);
    compd(std::move(in_args), ncvslideio::gapi::bind(out_args));
    // ! [bind after deserialization]
}

static void bind_deserialization_example()
{
    // ! [bind before serialization]
    std::vector<ncvslideio::GRunArgP> graph_outs;
    ncvslideio::GRunArgs out_args;

    for (auto &&out : graph_outs) {
        out_args.emplace_back(ncvslideio::gapi::bind(out));
    }
    const auto sargsout = ncvslideio::gapi::serialize(out_args);
    // ! [bind before serialization]
}

struct SimpleCustomType {
    bool val;
    bool operator==(const SimpleCustomType& other) const {
        return val == other.val;
    }
};

struct SimpleCustomType2 {
    int val;
    std::string name;
    std::vector<float> vec;
    std::map<int, uint64_t> mmap;
    bool operator==(const SimpleCustomType2& other) const {
        return val == other.val && name == other.name &&
               vec == other.vec && mmap == other.mmap;
    }
};

// ! [S11N usage]
namespace ncvslideio {
namespace gapi {
namespace s11n {
namespace detail {
template<> struct S11N<SimpleCustomType> {
    static void serialize(IOStream &os, const SimpleCustomType &p) {
        os << p.val;
    }
    static SimpleCustomType deserialize(IIStream &is) {
        SimpleCustomType p;
        is >> p.val;
        return p;
    }
};

template<> struct S11N<SimpleCustomType2> {
    static void serialize(IOStream &os, const SimpleCustomType2 &p) {
        os << p.val << p.name << p.vec << p.mmap;
    }
    static SimpleCustomType2 deserialize(IIStream &is) {
        SimpleCustomType2 p;
        is >> p.val >> p.name >> p.vec >> p.mmap;
        return p;
    }
};
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace ncvslideio
// ! [S11N usage]

namespace ncvslideio {
namespace detail {
template<> struct CompileArgTag<SimpleCustomType> {
    static const char* tag() {
        return "org.opencv.test.simple_custom_type";
    }
};

template<> struct CompileArgTag<SimpleCustomType2> {
    static const char* tag() {
        return "org.opencv.test.simple_custom_type_2";
    }
};
} // namespace detail
} // namespace ncvslideio

static void s11n_example()
{
    SimpleCustomType  customVar1 { false };
    SimpleCustomType2 customVar2 { 1248, "World", {1280, 720, 640, 480},
                                   { {5, 32434142342}, {7, 34242432} } };

    std::vector<char> sArgs = ncvslideio::gapi::serialize(
        ncvslideio::compile_args(customVar1, customVar2));

    ncvslideio::GCompileArgs dArgs = ncvslideio::gapi::deserialize<ncvslideio::GCompileArgs,
                                                   SimpleCustomType,
                                                   SimpleCustomType2>(sArgs);

    SimpleCustomType  dCustomVar1 = ncvslideio::gapi::getCompileArg<SimpleCustomType>(dArgs).value();
    SimpleCustomType2 dCustomVar2 = ncvslideio::gapi::getCompileArg<SimpleCustomType2>(dArgs).value();

    (void) dCustomVar1;
    (void) dCustomVar2;
}

G_TYPED_KERNEL(IAdd, <ncvslideio::GMat(ncvslideio::GMat)>, "test.custom.add") {
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in) { return in; }
};
G_TYPED_KERNEL(IFilter2D, <ncvslideio::GMat(ncvslideio::GMat)>, "test.custom.filter2d") {
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in) { return in; }
};
G_TYPED_KERNEL(IRGB2YUV, <ncvslideio::GMat(ncvslideio::GMat)>, "test.custom.add") {
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in) { return in; }
};
GAPI_OCV_KERNEL(CustomAdd,      IAdd)      { static void run(ncvslideio::Mat, ncvslideio::Mat &) {} };
GAPI_OCV_KERNEL(CustomFilter2D, IFilter2D) { static void run(ncvslideio::Mat, ncvslideio::Mat &) {} };
GAPI_OCV_KERNEL(CustomRGB2YUV,  IRGB2YUV)  { static void run(ncvslideio::Mat, ncvslideio::Mat &) {} };

int main(int argc, char *argv[])
{
    if (argc < 3)
        return -1;

    ncvslideio::Mat input = ncvslideio::imread(argv[1]);
    ncvslideio::Mat output;

    {
    //! [graph_def]
    ncvslideio::GMat in;
    ncvslideio::GMat gx = ncvslideio::gapi::Sobel(in, CV_32F, 1, 0);
    ncvslideio::GMat gy = ncvslideio::gapi::Sobel(in, CV_32F, 0, 1);
    ncvslideio::GMat g  = ncvslideio::gapi::sqrt(ncvslideio::gapi::mul(gx, gx) + ncvslideio::gapi::mul(gy, gy));
    ncvslideio::GMat out = ncvslideio::gapi::convertTo(g, CV_8U);
    //! [graph_def]

    //! [graph_decl_apply]
    //! [graph_cap_full]
    ncvslideio::GComputation sobelEdge(ncvslideio::GIn(in), ncvslideio::GOut(out));
    //! [graph_cap_full]
    sobelEdge.apply(input, output);
    //! [graph_decl_apply]

    //! [apply_with_param]
    ncvslideio::GKernelPackage kernels = ncvslideio::gapi::combine
        (ncvslideio::gapi::core::fluid::kernels(),
         ncvslideio::gapi::imgproc::fluid::kernels());
    sobelEdge.apply(input, output, ncvslideio::compile_args(kernels));
    //! [apply_with_param]

    //! [graph_cap_sub]
    ncvslideio::GComputation sobelEdgeSub(ncvslideio::GIn(gx, gy), ncvslideio::GOut(out));
    //! [graph_cap_sub]
    }
    //! [graph_gen]
    ncvslideio::GComputation sobelEdgeGen([](){
            ncvslideio::GMat in;
            ncvslideio::GMat gx = ncvslideio::gapi::Sobel(in, CV_32F, 1, 0);
            ncvslideio::GMat gy = ncvslideio::gapi::Sobel(in, CV_32F, 0, 1);
            ncvslideio::GMat g  = ncvslideio::gapi::sqrt(ncvslideio::gapi::mul(gx, gx) + ncvslideio::gapi::mul(gy, gy));
            ncvslideio::GMat out = ncvslideio::gapi::convertTo(g, CV_8U);
            return ncvslideio::GComputation(in, out);
        });
    //! [graph_gen]

    ncvslideio::imwrite(argv[2], output);

    //! [kernels_snippet]
    ncvslideio::GKernelPackage pkg = ncvslideio::gapi::kernels
        < CustomAdd
        , CustomFilter2D
        , CustomRGB2YUV
        >();
    //! [kernels_snippet]

    // Just call typed example with no input/output - avoid warnings about
    // unused functions
    typed_example();
    gscalar_example();
    bind_serialization_example();
    bind_deserialization_example();
    s11n_example();
    return 0;
}
