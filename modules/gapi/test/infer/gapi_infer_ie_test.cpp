// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2023 Intel Corporation

#include "../test_precomp.hpp"

#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE < 2023010000

#include <stdexcept>
#include <mutex>
#include <condition_variable>

#include <inference_engine.hpp>

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include "backends/ie/util.hpp"
#include "backends/ie/giebackend/giewrapper.hpp"

#ifdef HAVE_NGRAPH
#if defined(__clang__)  // clang or MSVC clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
# if _MSC_VER < 1910
#  pragma warning(disable:4268) // Disable warnings of ngraph. OpenVINO recommends to use MSVS 2019.
#  pragma warning(disable:4800)
# endif
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#endif

namespace opencv_test
{
namespace {

class TestMediaBGR final: public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_mat;
    using Cb = ncvslideio::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaBGR(ncvslideio::Mat m, Cb cb = [](){})
        : m_mat(m), m_cb(cb) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::BGR, ncvslideio::Size(m_mat.cols, m_mat.rows)};
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss), Cb{m_cb});
    }
    ncvslideio::util::any blobParams() const override {
#if INF_ENGINE_RELEASE > 2023000000
        // NB: blobParams() shouldn't be used in tests
        // if OpenVINO versions is higher than 2023.0
        GAPI_Assert(false && "NV12 feature has been deprecated in OpenVINO 1.0 API.");
#else
        return std::make_pair<InferenceEngine::TensorDesc,
                              InferenceEngine::ParamMap>({IE::Precision::U8,
                                                          {1, 3, 300, 300},
                                                          IE::Layout::NCHW},
                                                         {{"HELLO", 42},
                                                          {"COLOR_FORMAT",
                                                           InferenceEngine::ColorFormat::NV12}});
#endif // INF_ENGINE_RELEASE > 2023000000
    }
};

class TestMediaNV12 final: public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_y;
    ncvslideio::Mat m_uv;
public:
    TestMediaNV12(ncvslideio::Mat y, ncvslideio::Mat uv) : m_y(y), m_uv(uv) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::NV12, ncvslideio::Size(m_y.cols, m_y.rows)};
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        ncvslideio::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
    }
};

// FIXME: taken from DNN module
static void initDLDTDataPath()
{
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
        const char* omzDataPath = getenv("OPENCV_OPEN_MODEL_ZOO_DATA_PATH");
        if (omzDataPath)
            cvtest::addDataSearchPath(omzDataPath);
        const char* dnnDataPath = getenv("OPENCV_DNN_TEST_DATA_PATH");
        if (dnnDataPath) {
            // Add the dnnDataPath itself - G-API is using some images there directly
            cvtest::addDataSearchPath(dnnDataPath);
            cvtest::addDataSearchPath(dnnDataPath + std::string("/omz_intel_models"));
        }
        initialized = true;
    }
#endif // WINRT
}

#if INF_ENGINE_RELEASE >= 2020010000
static const std::string SUBDIR = "intel/age-gender-recognition-retail-0013/FP32/";
#else
static const std::string SUBDIR = "Retail/object_attributes/age_gender/dldt/";
#endif

// FIXME: taken from the DNN module
void normAssert(ncvslideio::InputArray ref, ncvslideio::InputArray test,
                const char *comment /*= ""*/,
                double l1 = 0.00001, double lInf = 0.0001)
{
    double normL1 = cvtest::norm(ref, test, ncvslideio::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    double normInf = cvtest::norm(ref, test, ncvslideio::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

namespace IE = InferenceEngine;

void setNetParameters(IE::CNNNetwork& net, bool is_nv12 = false) {
    auto ii = net.getInputsInfo().at("data");
    ii->setPrecision(IE::Precision::U8);
    ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
    if (is_nv12) {
#if INF_ENGINE_RELEASE > 2023000000
        // NB: NV12 feature shouldn't be used in tests
        // if OpenVINO versions is higher than 2023.0
        GAPI_Assert(false && "NV12 feature has been deprecated in OpenVINO 1.0 API.");
#else
        ii->getPreProcess().setColorFormat(IE::ColorFormat::NV12);
#endif // INF_ENGINE_RELEASE > 2023000000
    }
}

bool checkDeviceIsAvailable(const std::string& device) {
    const static auto available_devices = [&](){
        auto devices = ncvslideio::gimpl::ie::wrap::getCore().GetAvailableDevices();
        return std::unordered_set<std::string>{devices.begin(), devices.end()};
    }();
    return available_devices.find(device) != available_devices.end();
}

void skipIfDeviceNotAvailable(const std::string& device) {
    if (!checkDeviceIsAvailable(device)) {
        throw SkipTestException("Device: " + device + " isn't available!");
    }
}

void compileBlob(const ncvslideio::gapi::ie::detail::ParamDesc& params,
                 const std::string&                     output,
                 const IE::Precision&                   ip) {
    auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
    auto net    = ncvslideio::gimpl::ie::wrap::readNetwork(params);
    for (auto&& ii : net.getInputsInfo()) {
        ii.second->setPrecision(ip);
    }
    auto this_network = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
    std::ofstream out_file{output, std::ios::out | std::ios::binary};
    GAPI_Assert(out_file.is_open());
    this_network.Export(out_file);
}

std::string compileAgeGenderBlob(const std::string& device) {
    const static std::string blob_path = [&](){
        ncvslideio::gapi::ie::detail::ParamDesc params;
        const std::string model_name = "age-gender-recognition-retail-0013";
        const std::string output  = model_name + ".blob";
        params.model_path   = findDataFile(SUBDIR + model_name + ".xml", false);
        params.weights_path = findDataFile(SUBDIR + model_name + ".bin", false);
        params.device_id    = device;
        compileBlob(params, output, IE::Precision::U8);
        return output;
    }();
    return blob_path;
}

} // anonymous namespace

// TODO: Probably DNN/IE part can be further parametrized with a template
// NOTE: here ".." is used to leave the default "gapi/" search scope
TEST(TestAgeGenderIE, InferBasicTensor)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    ncvslideio::Mat gapi_age, gapi_gender;

    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();

        const auto &iedims = net.getInputsInfo().begin()->second->getTensorDesc().getDims();
              auto  cvdims = ncvslideio::gapi::ie::util::to_ocv(iedims);
        in_mat.create(cvdims, CV_32F);
        ncvslideio::randu(in_mat, -1, 1);

        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(TestAgeGenderIE, InferBasicImage)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // FIXME: Ideally it should be an image from disk
    // ncvslideio::Mat in_mat = ncvslideio::imread(findDataFile("grace_hopper_227.png"));
    ncvslideio::Mat in_mat(ncvslideio::Size(320, 240), CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

struct InferWithReshape: public ::testing::Test {
    ncvslideio::gapi::ie::detail::ParamDesc params;
    ncvslideio::Mat m_in_mat;
    std::vector<ncvslideio::Rect> m_roi_list;
    std::vector<size_t> reshape_dims;
    std::vector<ncvslideio::Mat> m_out_ie_ages;
    std::vector<ncvslideio::Mat> m_out_ie_genders;
    std::vector<ncvslideio::Mat> m_out_gapi_ages;
    std::vector<ncvslideio::Mat> m_out_gapi_genders;
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    InferenceEngine::CNNNetwork net;
    InferenceEngine::Core plugin;

    void SetUp() {
        // FIXME: it must be ncvslideio::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = ncvslideio::Mat(ncvslideio::Size(320, 240), CV_8UC3);
        ncvslideio::randu(m_in_mat, 0, 255);

        m_out_gapi_ages.resize(1);
        m_out_gapi_genders.resize(1);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
            ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
        };

        // New dimensions for "data" input
        reshape_dims = {1, 3, 70, 70};

        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);

        params.device_id = "CPU";

        plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        net    = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        net.reshape({{"data", reshape_dims}});
    }

    void inferROIs(IE::Blob::Ptr blob) {
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : m_roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            infer_request.SetBlob("data", IE::make_shared_blob(blob, ie_rc));
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    void infer(ncvslideio::Mat& in, const bool with_roi = false) {
        if (!with_roi) {
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in));
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        } else {
            auto frame_blob = ncvslideio::gapi::ie::util::to_ie(in);
            inferROIs(frame_blob);
        }
    }

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        GAPI_Assert(!m_out_gapi_ages.empty());
        ASSERT_EQ(m_out_gapi_genders.size(), m_out_gapi_ages.size());
        ASSERT_EQ(m_out_gapi_ages.size(), m_out_ie_ages.size());
        ASSERT_EQ(m_out_gapi_genders.size(), m_out_ie_genders.size());

        const size_t size = m_out_gapi_ages.size();
        for (size_t i = 0; i < size; ++i) {
            normAssert(m_out_ie_ages   [i], m_out_gapi_ages   [i], "Test age output");
            normAssert(m_out_ie_genders[i], m_out_gapi_genders[i], "Test gender output");
        }
    }
}; // InferWithReshape

struct InferWithReshapeNV12: public InferWithReshape {
    ncvslideio::Mat m_in_uv;
    ncvslideio::Mat m_in_y;
    void SetUp() {
        InferWithReshape::SetUp();
        ncvslideio::Size sz{320, 240};
        m_in_y = ncvslideio::Mat{sz, CV_8UC1};
        ncvslideio::randu(m_in_y, 0, 255);
        m_in_uv = ncvslideio::Mat{sz / 2, CV_8UC2};
        ncvslideio::randu(m_in_uv, 0, 255);
// NB: NV12 feature shouldn't be used in tests
// if OpenVINO versions is higher than 2023.0
#if INF_ENGINE_RELEASE <= 2023000000
        setNetParameters(net, true);
        net.reshape({{"data", reshape_dims}});
        auto frame_blob = ncvslideio::gapi::ie::util::to_ie(m_in_y, m_in_uv);
        inferROIs(frame_blob);
#endif // INF_ENGINE_RELEASE <= 2023000000
    }
};

struct ROIList: public ::testing::Test {
    ncvslideio::gapi::ie::detail::ParamDesc params;

    ncvslideio::Mat m_in_mat;
    std::vector<ncvslideio::Rect> m_roi_list;

    std::vector<ncvslideio::Mat> m_out_ie_ages;
    std::vector<ncvslideio::Mat> m_out_ie_genders;

    std::vector<ncvslideio::Mat> m_out_gapi_ages;
    std::vector<ncvslideio::Mat> m_out_gapi_genders;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        params.device_id = "CPU";

        // FIXME: it must be ncvslideio::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = ncvslideio::Mat(ncvslideio::Size(320, 240), CV_8UC3);
        ncvslideio::randu(m_in_mat, 0, 255);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
            ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
        };

        // Load & run IE network
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = ncvslideio::gapi::ie::util::to_ie(m_in_mat);

            for (auto &&rc : m_roi_list) {
                const auto ie_rc = IE::ROI {
                    0u
                    , static_cast<std::size_t>(rc.x)
                    , static_cast<std::size_t>(rc.y)
                    , static_cast<std::size_t>(rc.width)
                    , static_cast<std::size_t>(rc.height)
                };
                infer_request.SetBlob("data", IE::make_shared_blob(frame_blob, ie_rc));
                infer_request.Infer();

                using namespace ncvslideio::gapi::ie::util;
                m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
    } // ROIList()

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        ASSERT_EQ(2u, m_out_ie_ages.size());
        ASSERT_EQ(2u, m_out_ie_genders.size());
        ASSERT_EQ(2u, m_out_gapi_ages.size());
        ASSERT_EQ(2u, m_out_gapi_genders.size());

        normAssert(m_out_ie_ages   [0], m_out_gapi_ages   [0], "0: Test age output");
        normAssert(m_out_ie_genders[0], m_out_gapi_genders[0], "0: Test gender output");
        normAssert(m_out_ie_ages   [1], m_out_gapi_ages   [1], "1: Test age output");
        normAssert(m_out_ie_genders[1], m_out_gapi_genders[1], "1: Test gender output");
    }
}; // ROIList

struct ROIListNV12: public ::testing::Test {
    ncvslideio::gapi::ie::detail::ParamDesc params;

    ncvslideio::Mat m_in_uv;
    ncvslideio::Mat m_in_y;
    std::vector<ncvslideio::Rect> m_roi_list;

    std::vector<ncvslideio::Mat> m_out_ie_ages;
    std::vector<ncvslideio::Mat> m_out_ie_genders;

    std::vector<ncvslideio::Mat> m_out_gapi_ages;
    std::vector<ncvslideio::Mat> m_out_gapi_genders;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        params.device_id = "CPU";

        ncvslideio::Size sz{320, 240};
        m_in_y = ncvslideio::Mat{sz, CV_8UC1};
        ncvslideio::randu(m_in_y, 0, 255);
        m_in_uv = ncvslideio::Mat{sz / 2, CV_8UC2};
        ncvslideio::randu(m_in_uv, 0, 255);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
            ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
        };

// NB: NV12 feature shouldn't be used in tests
// if OpenVINO versions is higher than 2023.0
#if INF_ENGINE_RELEASE <= 2023000000
        {
            // Load & run IE network
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net, true);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = ncvslideio::gapi::ie::util::to_ie(m_in_y, m_in_uv);

            for (auto &&rc : m_roi_list) {
                const auto ie_rc = IE::ROI {
                    0u
                    , static_cast<std::size_t>(rc.x)
                    , static_cast<std::size_t>(rc.y)
                    , static_cast<std::size_t>(rc.width)
                    , static_cast<std::size_t>(rc.height)
                };
                infer_request.SetBlob("data", IE::make_shared_blob(frame_blob, ie_rc));
                infer_request.Infer();

                using namespace ncvslideio::gapi::ie::util;
                m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
#endif // INF_ENGINE_RELEASE <= 2023000000
    } // ROIList()

    void validate() {
#if INF_ENGINE_RELEASE <= 2023000000
        // Validate with IE itself (avoid DNN module dependency here)
        ASSERT_EQ(2u, m_out_ie_ages.size());
        ASSERT_EQ(2u, m_out_ie_genders.size());
        ASSERT_EQ(2u, m_out_gapi_ages.size());
        ASSERT_EQ(2u, m_out_gapi_genders.size());

        normAssert(m_out_ie_ages   [0], m_out_gapi_ages   [0], "0: Test age output");
        normAssert(m_out_ie_genders[0], m_out_gapi_genders[0], "0: Test gender output");
        normAssert(m_out_ie_ages   [1], m_out_gapi_ages   [1], "1: Test age output");
        normAssert(m_out_ie_genders[1], m_out_gapi_genders[1], "1: Test gender output");
#else
        GAPI_Assert(false && "Reference hasn't been calculated because"
                             " NV12 feature has been deprecated.");
#endif // INF_ENGINE_RELEASE <= 2023000000
    }
};

struct SingleROI: public ::testing::Test {
    ncvslideio::gapi::ie::detail::ParamDesc params;

    ncvslideio::Mat m_in_mat;
    ncvslideio::Rect m_roi;

    ncvslideio::Mat m_out_gapi_age;
    ncvslideio::Mat m_out_gapi_gender;

    ncvslideio::Mat m_out_ie_age;
    ncvslideio::Mat m_out_ie_gender;

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        params.device_id = "CPU";

        // FIXME: it must be ncvslideio::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = ncvslideio::Mat(ncvslideio::Size(320, 240), CV_8UC3);
        ncvslideio::randu(m_in_mat, 0, 255);

        m_roi = ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();

            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(m_roi.x)
                , static_cast<std::size_t>(m_roi.y)
                , static_cast<std::size_t>(m_roi.width)
                , static_cast<std::size_t>(m_roi.height)
            };

            IE::Blob::Ptr roi_blob = IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(m_in_mat), ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();

            using namespace ncvslideio::gapi::ie::util;
            m_out_ie_age    = to_ocv(infer_request.GetBlob("age_conv3")).clone();
            m_out_ie_gender = to_ocv(infer_request.GetBlob("prob")).clone();
        }
    }

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(m_out_ie_age   , m_out_gapi_age   , "Test age output");
        normAssert(m_out_ie_gender, m_out_gapi_gender, "Test gender output");
    }
};

struct SingleROINV12: public ::testing::Test {
    ncvslideio::gapi::ie::detail::ParamDesc params;

    ncvslideio::Mat m_in_y;
    ncvslideio::Mat m_in_uv;
    ncvslideio::Rect m_roi;

    ncvslideio::Mat m_out_gapi_age;
    ncvslideio::Mat m_out_gapi_gender;

    ncvslideio::Mat m_out_ie_age;
    ncvslideio::Mat m_out_ie_gender;

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        params.device_id = "CPU";

        ncvslideio::Size sz{320, 240};
        m_in_y = ncvslideio::Mat{sz, CV_8UC1};
        ncvslideio::randu(m_in_y, 0, 255);
        m_in_uv = ncvslideio::Mat{sz / 2, CV_8UC2};
        ncvslideio::randu(m_in_uv, 0, 255);

        m_roi = ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

// NB: NV12 feature shouldn't be used in tests
// if OpenVINO versions is higher than 2023.0
#if INF_ENGINE_RELEASE <= 2023000000
        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net, /* NV12 */ true);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto blob = ncvslideio::gapi::ie::util::to_ie(m_in_y, m_in_uv);

            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(m_roi.x)
                , static_cast<std::size_t>(m_roi.y)
                , static_cast<std::size_t>(m_roi.width)
                , static_cast<std::size_t>(m_roi.height)
            };

            IE::Blob::Ptr roi_blob = IE::make_shared_blob(blob, ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();

            using namespace ncvslideio::gapi::ie::util;
            m_out_ie_age    = to_ocv(infer_request.GetBlob("age_conv3")).clone();
            m_out_ie_gender = to_ocv(infer_request.GetBlob("prob")).clone();
        }
#endif // INF_ENGINE_RELEASE <= 2023000000
    }

    void validate() {
#if INF_ENGINE_RELEASE <= 2023000000
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(m_out_ie_age   , m_out_gapi_age   , "Test age output");
        normAssert(m_out_ie_gender, m_out_gapi_gender, "Test gender output");
#else
        GAPI_Assert(false && "Reference hasn't been calculated because"
                             " NV12 feature has been deprecated.");
#endif
    }
};

TEST_F(ROIList, TestInfer)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
            ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(ROIList, TestInfer2)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST(DISABLED_TestTwoIENNPipeline, InferBasicImage)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc AGparams;
    AGparams.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    AGparams.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    AGparams.device_id = "MYRIAD";

    // FIXME: Ideally it should be an image from disk
    // ncvslideio::Mat in_mat = ncvslideio::imread(findDataFile("grace_hopper_227.png"));
    ncvslideio::Mat in_mat(ncvslideio::Size(320, 240), CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age1, gapi_gender1, gapi_age2, gapi_gender2;

    // Load & run IE network
    IE::Blob::Ptr ie_age1, ie_gender1, ie_age2, ie_gender2;
    {
        auto AGplugin1         = ncvslideio::gimpl::ie::wrap::getPlugin(AGparams);
        auto AGnet1            = ncvslideio::gimpl::ie::wrap::readNetwork(AGparams);
        setNetParameters(AGnet1);
        auto AGplugin_network1 = ncvslideio::gimpl::ie::wrap::loadNetwork(AGplugin1, AGnet1, AGparams);
        auto AGinfer_request1  = AGplugin_network1.CreateInferRequest();
        AGinfer_request1.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        AGinfer_request1.Infer();
        ie_age1    = AGinfer_request1.GetBlob("age_conv3");
        ie_gender1 = AGinfer_request1.GetBlob("prob");

        auto AGplugin2         = ncvslideio::gimpl::ie::wrap::getPlugin(AGparams);
        auto AGnet2            = ncvslideio::gimpl::ie::wrap::readNetwork(AGparams);
        setNetParameters(AGnet2);
        auto AGplugin_network2 = ncvslideio::gimpl::ie::wrap::loadNetwork(AGplugin2, AGnet2, AGparams);
        auto AGinfer_request2     = AGplugin_network2.CreateInferRequest();
        AGinfer_request2.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        AGinfer_request2.Infer();
        ie_age2    = AGinfer_request2.GetBlob("age_conv3");
        ie_gender2 = AGinfer_request2.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender1, <AGInfo(ncvslideio::GMat)>,   "test-age-gender1");
    G_API_NET(AgeGender2, <AGInfo(ncvslideio::GMat)>,   "test-age-gender2");
    ncvslideio::GMat in;
    ncvslideio::GMat age1, gender1;
    std::tie(age1, gender1) = ncvslideio::gapi::infer<AgeGender1>(in);

    ncvslideio::GMat age2, gender2;
    // FIXME: "Multi-node inference is not supported!", workarounded 'till enabling proper tools
    std::tie(age2, gender2) = ncvslideio::gapi::infer<AgeGender2>(ncvslideio::gapi::copy(in));
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age1, gender1, age2, gender2));

    auto age_net1 = ncvslideio::gapi::ie::Params<AgeGender1> {
        AGparams.model_path, AGparams.weights_path, AGparams.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    auto age_net2 = ncvslideio::gapi::ie::Params<AgeGender2> {
        AGparams.model_path, AGparams.weights_path, AGparams.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age1, gapi_gender1, gapi_age2, gapi_gender2),
               ncvslideio::compile_args(ncvslideio::gapi::networks(age_net1, age_net2)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age1),    gapi_age1,    "Test age output 1");
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender1), gapi_gender1, "Test gender output 1");
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age2),    gapi_age2,    "Test age output 2");
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender2), gapi_gender2, "Test gender output 2");
}

TEST(TestAgeGenderIE, GenericInfer)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Mat in_mat(ncvslideio::Size(320, 240), CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net    = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    ncvslideio::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", inputs);

    auto age    = outputs.at("age_conv3");
    auto gender = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id};

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(TestAgeGenderIE, InvalidConfigGeneric)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    std::string device_id    = "CPU";

    // Configure & run G-API
    ncvslideio::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic>{
        "age-gender-generic", model_path, weights_path, device_id
    }.pluginConfig({{"unsupported_config", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{320, 240}},
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, CPUConfigGeneric)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    std::string device_id    = "CPU";

    // Configure & run G-API
    ncvslideio::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> {
        "age-gender-generic", model_path, weights_path, device_id
    }.pluginConfig({{IE::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                     IE::PluginConfigParams::CPU_THROUGHPUT_NUMA}});

    EXPECT_NO_THROW(comp.compile(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{320, 240}},
                    ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, InvalidConfig)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    std::string device_id    = "CPU";

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        model_path, weights_path, device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .pluginConfig({{"unsupported_config", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{320, 240}},
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, CPUConfig)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    std::string device_id    = "CPU";

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        model_path, weights_path, device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .pluginConfig({{IE::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                     IE::PluginConfigParams::CPU_THROUGHPUT_NUMA}});

    EXPECT_NO_THROW(comp.compile(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{320, 240}},
                    ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST_F(ROIList, MediaInputBGR)
{
    initDLDTDataPath();

    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(ROIListNV12, MediaInputNV12)
{
    initDLDTDataPath();

    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi_list),
                     ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST(TestAgeGenderIE, MediaInputNV12)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_y_mat(sz, CV_8UC1);
    ncvslideio::randu(in_y_mat, 0, 255);
    ncvslideio::Mat in_uv_mat(sz / 2, CV_8UC2);
    ncvslideio::randu(in_uv_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

// NB: NV12 feature shouldn't be used in tests
// if OpenVINO versions is higher than 2023.0
#if INF_ENGINE_RELEASE <= 2023000000
    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net, true);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }
#endif

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame), ncvslideio::gout(gapi_age, gapi_gender),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST(TestAgeGenderIE, MediaInputBGR)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(in_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(frame), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(InferROI, MediaInputBGR)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;
    ncvslideio::Rect rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_mat), ie_rc);
        infer_request.SetBlob("data", roi_blob);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(in_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(frame, rect), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(InferROI, MediaInputNV12)
{
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Size sz{320, 240};
    auto in_y_mat = ncvslideio::Mat{sz, CV_8UC1};
    ncvslideio::randu(in_y_mat, 0, 255);
    auto in_uv_mat = ncvslideio::Mat{sz / 2, CV_8UC2};
    ncvslideio::randu(in_uv_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;
    ncvslideio::Rect rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

// NB: NV12 feature shouldn't be used in tests
// if OpenVINO versions is higher than 2023.0
#if INF_ENGINE_RELEASE <= 2023000000
    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net, true);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
        infer_request.SetBlob("data", roi_blob);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }
#endif

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, rect), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, rect), ncvslideio::gout(gapi_age, gapi_gender),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST_F(ROIList, Infer2MediaInputBGR)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(ROIListNV12, Infer2MediaInputNV12)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi_list),
                     ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST_F(SingleROI, GenericInfer)
{
    // Configure & run G-API
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(ncvslideio::gin(m_in_mat, m_roi), ncvslideio::gout(m_out_gapi_age, m_out_gapi_gender),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(SingleROI, GenericInferMediaBGR)
{
    // Configure & run G-API
    ncvslideio::GFrame in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(ncvslideio::gin(frame, m_roi), ncvslideio::gout(m_out_gapi_age, m_out_gapi_gender),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(SingleROINV12, GenericInferMediaNV12)
{
    // Configure & run G-API
    ncvslideio::GFrame in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi), ncvslideio::gout(m_out_gapi_age, m_out_gapi_gender),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi),
                     ncvslideio::gout(m_out_gapi_age, m_out_gapi_gender),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST_F(ROIList, GenericInfer)
{
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
            ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(ROIList, GenericInferMediaBGR)
{
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(ncvslideio::gin(frame, m_roi_list),
            ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(ROIListNV12, GenericInferMediaNV12)
{
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    // NB: NV12 feature has been deprecated in OpenVINO versions higher
    // than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi_list),
                     ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST_F(ROIList, GenericInfer2)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    GInferListInputs list;
    list["data"] = rr;

    auto outputs = ncvslideio::gapi::infer2<ncvslideio::gapi::Generic>("age-gender-generic", in, list);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
            ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(ROIList, GenericInfer2MediaInputBGR)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    GInferListInputs inputs;
    inputs["data"] = rr;

    auto outputs = ncvslideio::gapi::infer2<ncvslideio::gapi::Generic>("age-gender-generic", in, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(ncvslideio::gin(frame, m_roi_list),
            ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(ROIListNV12, GenericInfer2MediaInputNV12)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    GInferListInputs inputs;
    inputs["data"] = rr;

    auto outputs = ncvslideio::gapi::infer2<ncvslideio::gapi::Generic>("age-gender-generic", in, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi_list),
                     ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST(Infer, SetInvalidNumberOfRequests)
{
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::gapi::ie::Params<AgeGender> pp{"model", "weights", "device"};

    EXPECT_ANY_THROW(pp.cfgNumRequests(0u));
}

TEST(Infer, TestStreamingInfer)
{
    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip InferROI.TestStreamingInfer as it hangs sporadically");

    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    ncvslideio::Mat gapi_age, gapi_gender;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;

    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);


    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource<ncvslideio::gapi::wip::GCaptureSource>(filepath);

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_age, gapi_gender)))
    {
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();

            infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
            infer_request.Infer();
            ie_age    = infer_request.GetBlob("age_conv3");
            ie_gender = infer_request.GetBlob("prob");
        }
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferROI, TestStreamingInfer)
{
    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip InferROI.TestStreamingInfer as it hangs sporadically");

    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    ncvslideio::Mat gapi_age, gapi_gender;
    ncvslideio::Rect rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GMat age, gender;

    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);


    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource(
            ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(filepath), rect));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_age, gapi_gender)))
    {
        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rect.x)
                , static_cast<std::size_t>(rect.y)
                , static_cast<std::size_t>(rect.width)
                , static_cast<std::size_t>(rect.height)
            };
            IE::Blob::Ptr roi_blob = IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_mat), ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();
            ie_age    = infer_request.GetBlob("age_conv3");
            ie_gender = infer_request.GetBlob("prob");
        }
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferList, TestStreamingInfer)
{
    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip InferList.TestStreamingInfer as it hangs sporadically");

    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    std::vector<ncvslideio::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::Rect> roi;
    ncvslideio::GArray<GMat> age, gender;

    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource(
            ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_ages, gapi_genders)))
    {
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = ncvslideio::gapi::ie::util::to_ie(in_mat);

            for (auto &&rc : roi_list) {
                const auto ie_rc = IE::ROI {
                    0u
                    , static_cast<std::size_t>(rc.x)
                    , static_cast<std::size_t>(rc.y)
                    , static_cast<std::size_t>(rc.width)
                    , static_cast<std::size_t>(rc.height)
                };
                infer_request.SetBlob("data", IE::make_shared_blob(frame_blob, ie_rc));
                infer_request.Infer();

                using namespace ncvslideio::gapi::ie::util;
                ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ie_ages   [0], gapi_ages   [0], "0: Test age output");
        normAssert(ie_genders[0], gapi_genders[0], "0: Test gender output");
        normAssert(ie_ages   [1], gapi_ages   [1], "1: Test age output");
        normAssert(ie_genders[1], gapi_genders[1], "1: Test gender output");

        ie_ages.clear();
        ie_genders.clear();

        ++num_frames;
        cap >> in_mat;
    }
}

TEST(Infer2, TestStreamingInfer)
{
    if (cvtest::skipUnstableTests)
        throw SkipTestException("Skip InferROI.TestStreamingInfer as it hangs sporadically");

    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    std::vector<ncvslideio::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource(
            ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_ages, gapi_genders)))
    {
        {
            auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
            auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = ncvslideio::gapi::ie::util::to_ie(in_mat);

            for (auto &&rc : roi_list) {
                const auto ie_rc = IE::ROI {
                    0u
                    , static_cast<std::size_t>(rc.x)
                    , static_cast<std::size_t>(rc.y)
                    , static_cast<std::size_t>(rc.width)
                    , static_cast<std::size_t>(rc.height)
                };
                infer_request.SetBlob("data", IE::make_shared_blob(frame_blob, ie_rc));
                infer_request.Infer();

                using namespace ncvslideio::gapi::ie::util;
                ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ie_ages   [0], gapi_ages   [0], "0: Test age output");
        normAssert(ie_genders[0], gapi_genders[0], "0: Test gender output");
        normAssert(ie_ages   [1], gapi_ages   [1], "1: Test age output");
        normAssert(ie_genders[1], gapi_genders[1], "1: Test gender output");

        ie_ages.clear();
        ie_genders.clear();

        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferEmptyList, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    std::vector<ncvslideio::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    // NB: Empty list of roi
    std::vector<ncvslideio::Rect> roi_list;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::Rect> roi;
    ncvslideio::GArray<GMat> age, gender;

    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 1u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource(
            ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_ages, gapi_genders)))
    {
        EXPECT_TRUE(gapi_ages.empty());
        EXPECT_TRUE(gapi_genders.empty());
    }
}

TEST(Infer2EmptyList, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat;
    std::vector<ncvslideio::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    // NB: Empty list of roi
    std::vector<ncvslideio::Rect> roi_list;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);

    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 1u;

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    pipeline.setSource(
            ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(ncvslideio::gout(gapi_ages, gapi_genders)))
    {
        EXPECT_TRUE(gapi_ages.empty());
        EXPECT_TRUE(gapi_genders.empty());
    }
}

TEST_F(InferWithReshape, TestInfer)
{
    // IE code
    infer(m_in_mat);
    // G-API code
    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferInImage)
{
    // Input image already has 70x70 size
    ncvslideio::Mat rsz;
    ncvslideio::resize(m_in_mat, rsz, ncvslideio::Size(70, 70));
    // IE code
    infer(rsz);
    // G-API code
    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({"data"});
    // Reshape CNN input by input image size
    comp.apply(ncvslideio::gin(rsz), ncvslideio::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferForSingleLayer)
{
    // IE code
    infer(m_in_mat);
    // G-API code
    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgInputReshape("data", reshape_dims);
    comp.apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferList)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferList2)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(ncvslideio::gin(m_in_mat, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferListBGR)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshapeNV12, TestInferListYUV)
{
    // G-API code
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});

// NB: NV12 feature has been deprecated in OpenVINO versions higher
// than 2023.0 so G-API must throw error in that case.
#if INF_ENGINE_RELEASE <= 2023000000
    comp.apply(ncvslideio::gin(frame, m_roi_list),
               ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    // Validate
    validate();
#else
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(frame, m_roi_list),
                     ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
#endif
}

TEST_F(ROIList, CallInferMultipleTimes)
{
    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto cc = comp.compile(ncvslideio::descr_of(ncvslideio::gin(m_in_mat, m_roi_list)),
                           ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    for (int i = 0; i < 10; ++i) {
        cc(ncvslideio::gin(m_in_mat, m_roi_list), ncvslideio::gout(m_out_gapi_ages, m_out_gapi_genders));
    }

    validate();
}

#if INF_ENGINE_RELEASE <= 2023000000
TEST(IEFrameAdapter, blobParams)
{
    ncvslideio::Mat bgr = ncvslideio::Mat::eye(240, 320, CV_8UC3);
    ncvslideio::MediaFrame frame = ncvslideio::MediaFrame::Create<TestMediaBGR>(bgr);

    auto expected = std::make_pair(IE::TensorDesc{IE::Precision::U8, {1, 3, 300, 300},
                                                  IE::Layout::NCHW},
                                   IE::ParamMap{{"HELLO", 42}, {"COLOR_FORMAT",
                                                                IE::ColorFormat::NV12}});

    auto actual = ncvslideio::util::any_cast<decltype(expected)>(frame.blobParams());

    EXPECT_EQ(expected, actual);
}
#endif

namespace
{

struct Sync {
    std::mutex              m;
    std::condition_variable ncvslideio;
    int                     counter = 0;
};

class GMockMediaAdapter final: public ncvslideio::MediaFrame::IAdapter {
public:
    explicit GMockMediaAdapter(ncvslideio::Mat m, std::shared_ptr<Sync> sync)
        : m_mat(m), m_sync(sync) {
    }

    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::BGR, m_mat.size()};
    }

    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
    }

    ~GMockMediaAdapter() {
        {
            std::lock_guard<std::mutex> lk{m_sync->m};
            m_sync->counter--;
        }
        m_sync->ncvslideio.notify_one();
    }

private:
    ncvslideio::Mat               m_mat;
    std::shared_ptr<Sync> m_sync;
};

// NB: This source is needed to simulate real
// cases where the memory resources are limited.
// GMockSource(int limit) - accept the number of MediaFrames that
// the source can produce until resources are over.
class GMockSource : public ncvslideio::gapi::wip::IStreamSource {
public:
    explicit GMockSource(int limit)
        : m_limit(limit), m_mat(ncvslideio::Size(1920, 1080), CV_8UC3),
          m_sync(new Sync{}) {
        ncvslideio::randu(m_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    }

    bool pull(ncvslideio::gapi::wip::Data& data) {
        std::unique_lock<std::mutex> lk(m_sync->m);
        m_sync->counter++;
        // NB: Can't produce new frames until old ones are released.
        m_sync->ncvslideio.wait(lk, [this]{return m_sync->counter <= m_limit;});

        data = ncvslideio::MediaFrame::Create<GMockMediaAdapter>(m_mat, m_sync);
        return true;
    }

    GMetaArg descr_of() const override {
        return GMetaArg{ncvslideio::GFrameDesc{ncvslideio::MediaFormat::BGR, m_mat.size()}};
    }

private:
    int                   m_limit;
    ncvslideio::Mat               m_mat;
    std::shared_ptr<Sync> m_sync;
};

struct LimitedSourceInfer: public ::testing::Test {
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    LimitedSourceInfer()
        : comp([](){
            ncvslideio::GFrame in;
            ncvslideio::GMat age, gender;
            std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
            return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));
        }) {
        initDLDTDataPath();
    }

    GStreamingCompiled compileStreaming(int nireq) {
        ncvslideio::gapi::ie::detail::ParamDesc params;
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        params.device_id = "CPU";

        auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
            params.model_path, params.weights_path, params.device_id }
        .cfgOutputLayers({ "age_conv3", "prob" })
        .cfgNumRequests(nireq);

        return comp.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    }

    void run(const int max_frames, const int limit, const int nireq) {
        auto pipeline = compileStreaming(nireq);
        pipeline.setSource<GMockSource>(limit);
        pipeline.start();

        int num_frames = 0;
        while (num_frames != max_frames &&
               pipeline.pull(ncvslideio::gout(out_age, out_gender))) {
            ++num_frames;
        }
    }

    ncvslideio::GComputation comp;
    ncvslideio::Mat          out_age, out_gender;
};

} // anonymous namespace

TEST_F(LimitedSourceInfer, ReleaseFrame)
{
    constexpr int max_frames      = 50;
    constexpr int resources_limit = 1;
    constexpr int nireq           = 1;

    run(max_frames, resources_limit, nireq);
}

TEST_F(LimitedSourceInfer, ReleaseFrameAsync)
{
    constexpr int max_frames      = 50;
    constexpr int resources_limit = 4;
    constexpr int nireq           = 8;

    run(max_frames, resources_limit, nireq);
}

TEST(TestAgeGenderIE, InferWithBatch)
{
    initDLDTDataPath();

    constexpr int batch_size = 4;
    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    ncvslideio::Mat in_mat({batch_size, 3, 62, 62}, CV_8U);
    ncvslideio::randu(in_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        auto ii = net.getInputsInfo().at("data");
        ii->setPrecision(IE::Precision::U8);
        net.setBatchSize(batch_size);
        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgBatchSize(batch_size);

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

// NB: All tests below use preprocessing for "Import" networks
// passed as the last argument to SetBLob. This overload has
// been deprecated in OpenVINO 1.0 API.
#if INF_ENGINE_RELEASE <= 2023000000
TEST(ImportNetwork, Infer)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Mat in_mat(320, 240, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);
    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_mat), info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferNV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path= compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_y_mat(sz, CV_8UC1);
    ncvslideio::randu(in_y_mat, 0, 255);
    ncvslideio::Mat in_uv_mat(sz / 2, CV_8UC2);
    ncvslideio::randu(in_uv_mat, 0, 255);

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        info.setColorFormat(IE::ColorFormat::NV12);
        infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(ncvslideio::gin(frame), ncvslideio::gout(gapi_age, gapi_gender),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferROI)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Mat in_mat(320, 240, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);
    ncvslideio::Mat gapi_age, gapi_gender;
    ncvslideio::Rect rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_mat), ie_rc);
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        infer_request.SetBlob("data", roi_blob, info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(in_mat, rect), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferROINV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_y_mat(sz, CV_8UC1);
    ncvslideio::randu(in_y_mat, 0, 255);
    ncvslideio::Mat in_uv_mat(sz / 2, CV_8UC2);
    ncvslideio::randu(in_uv_mat, 0, 255);
    ncvslideio::Rect rect(ncvslideio::Point{64, 60}, ncvslideio::Size{96, 96});

    ncvslideio::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob =
            IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        info.setColorFormat(IE::ColorFormat::NV12);
        infer_request.SetBlob("data", roi_blob, info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GFrame in;
    ncvslideio::GOpaque<ncvslideio::Rect> roi;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(roi, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, roi), ncvslideio::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(frame, rect), ncvslideio::gout(gapi_age, gapi_gender),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferList)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Mat in_mat(320, 240, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);
    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };
    std::vector<ncvslideio::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(in_mat, roi_list), ncvslideio::gout(out_gapi_ages, out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferListNV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_y_mat(sz, CV_8UC1);
    ncvslideio::randu(in_y_mat, 0, 255);
    ncvslideio::Mat in_uv_mat(sz / 2, CV_8UC2);
    ncvslideio::randu(in_uv_mat, 0, 255);
    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };
    std::vector<ncvslideio::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            info.setColorFormat(IE::ColorFormat::NV12);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(rr, in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    comp.apply(ncvslideio::gin(frame, roi_list), ncvslideio::gout(out_gapi_ages, out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferList2)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Mat in_mat(320, 240, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);
    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };
    std::vector<ncvslideio::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(ncvslideio::gin(in_mat, roi_list), ncvslideio::gout(out_gapi_ages, out_gapi_genders),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferList2NV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    ncvslideio::Size sz{320, 240};
    ncvslideio::Mat in_y_mat(sz, CV_8UC1);
    ncvslideio::randu(in_y_mat, 0, 255);
    ncvslideio::Mat in_uv_mat(sz / 2, CV_8UC2);
    ncvslideio::randu(in_uv_mat, 0, 255);
    std::vector<ncvslideio::Rect> roi_list = {
        ncvslideio::Rect(ncvslideio::Point{64, 60}, ncvslideio::Size{ 96,  96}),
        ncvslideio::Rect(ncvslideio::Point{50, 32}, ncvslideio::Size{128, 160}),
    };
    std::vector<ncvslideio::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = ncvslideio::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(ncvslideio::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            info.setColorFormat(IE::ColorFormat::NV12);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace ncvslideio::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GArray<ncvslideio::Rect> rr;
    ncvslideio::GFrame in;
    ncvslideio::GArray<ncvslideio::GMat> age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer2<AgeGender>(in, rr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, rr), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    comp.apply(ncvslideio::gin(frame, roi_list), ncvslideio::gout(out_gapi_ages, out_gapi_genders),
            ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}
#endif

TEST(TestAgeGender, ThrowBlobAndInputPrecisionMismatch)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    // NB: Precision for inputs is U8.
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in, age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    ncvslideio::Mat in_mat(320, 240, CV_32FC3);
    ncvslideio::randu(in_mat, 0, 1);
    ncvslideio::Mat gapi_age, gapi_gender;

    // NB: Blob precision is U8, but user pass FP32 data, so exception will be thrown.
    // Now exception comes directly from IE, but since G-API has information
    // about data precision at the compile stage, consider the possibility of
    // throwing exception from there.
    EXPECT_ANY_THROW(comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
                     ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

#ifdef HAVE_NGRAPH

TEST(Infer, ModelWith2DInputs)
{
    const std::string model_name   = "ModelWith2DInputs";
    const std::string model_path   = model_name + ".xml";
    const std::string weights_path = model_name + ".bin";
    const std::string device_id    = "CPU";
    const int W                    = 10;
    const int H                    = 5;

    // NB: Define model with 2D inputs.
    auto in1 = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Type_t::u8,
        ngraph::Shape(std::vector<size_t>{(size_t)H, (size_t)W})
    );
    auto in2 = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Type_t::u8,
        ngraph::Shape(std::vector<size_t>{(size_t)H, (size_t)W})
    );
    auto result = std::make_shared<ngraph::op::v1::Add>(in1, in2);
    auto func   = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{result},
        ngraph::ParameterVector{in1, in2}
    );

    ncvslideio::Mat in_mat1(std::vector<int>{H, W}, CV_8U),
            in_mat2(std::vector<int>{H, W}, CV_8U),
            gapi_mat, ref_mat;

    ncvslideio::randu(in_mat1, 0, 100);
    ncvslideio::randu(in_mat2, 0, 100);
    ncvslideio::add(in_mat1, in_mat2, ref_mat, ncvslideio::noArray(), CV_32F);

    // Compile xml file
    IE::CNNNetwork(func).serialize(model_path);

    // Configure & run G-API
    ncvslideio::GMat g_in1, g_in2;
    ncvslideio::GInferInputs inputs;
    inputs[in1->get_name()] = g_in1;
    inputs[in2->get_name()] = g_in2;
    auto outputs = ncvslideio::gapi::infer<ncvslideio::gapi::Generic>(model_name, inputs);
    auto out = outputs.at(result->get_name());

    ncvslideio::GComputation comp(ncvslideio::GIn(g_in1, g_in2), ncvslideio::GOut(out));

    auto pp = ncvslideio::gapi::ie::Params<ncvslideio::gapi::Generic>(model_name,
                                                      model_path,
                                                      weights_path,
                                                      device_id);

    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(gapi_mat),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    normAssert(ref_mat, gapi_mat, "Test model output");
}

#endif // HAVE_NGRAPH

TEST(TestAgeGender, ThrowBlobAndInputPrecisionMismatchStreaming)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    // NB: Precision for inputs is U8.
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    ncvslideio::GMat in, age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    auto pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(age, gender))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    ncvslideio::Mat in_mat(320, 240, CV_32FC3);
    ncvslideio::randu(in_mat, 0, 1);
    ncvslideio::Mat gapi_age, gapi_gender;

    pipeline.setSource(ncvslideio::gin(in_mat));
    pipeline.start();

    // NB: Blob precision is U8, but user pass FP32 data, so exception will be thrown.
    // Now exception comes directly from IE, but since G-API has information
    // about data precision at the compile stage, consider the possibility of
    // throwing exception from there.
    for (int i = 0; i < 10; ++i) {
        EXPECT_ANY_THROW(pipeline.pull(ncvslideio::gout(gapi_age, gapi_gender)));
    }
}

struct AgeGenderInferTest: public ::testing::Test {
    ncvslideio::Mat m_in_mat;
    ncvslideio::Mat m_gapi_age;
    ncvslideio::Mat m_gapi_gender;

    ncvslideio::gimpl::ie::wrap::Plugin     m_plugin;
    IE::CNNNetwork                  m_net;
    ncvslideio::gapi::ie::detail::ParamDesc m_params;

    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    void SetUp() {
        initDLDTDataPath();
        m_params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        m_params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        m_params.device_id = "CPU";

        m_plugin = ncvslideio::gimpl::ie::wrap::getPlugin(m_params);
        m_net    = ncvslideio::gimpl::ie::wrap::readNetwork(m_params);
        setNetParameters(m_net);

        m_in_mat = ncvslideio::Mat(ncvslideio::Size(320, 240), CV_8UC3);
        ncvslideio::randu(m_in_mat, 0, 255);
    }

    ncvslideio::GComputation buildGraph() {
        ncvslideio::GMat in, age, gender;
        std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
        return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));
    }

    void validate() {
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(m_plugin, m_net, m_params);
            auto infer_request = this_network.CreateInferRequest();
            infer_request.SetBlob("data", ncvslideio::gapi::ie::util::to_ie(m_in_mat));
            infer_request.Infer();
            ie_age    = infer_request.GetBlob("age_conv3");
            ie_gender = infer_request.GetBlob("prob");
        }
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    m_gapi_age,    "Test age output"   );
        normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), m_gapi_gender, "Test gender output");
    }
};

TEST_F(AgeGenderInferTest, SyncExecution) {
    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        m_params.model_path, m_params.weights_path, m_params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgInferMode(ncvslideio::gapi::ie::InferMode::Sync);

    buildGraph().apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_gapi_age, m_gapi_gender),
                       ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    validate();
}

TEST_F(AgeGenderInferTest, ThrowSyncWithNireqNotEqualToOne) {
    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        m_params.model_path, m_params.weights_path, m_params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgInferMode(ncvslideio::gapi::ie::InferMode::Sync)
     .cfgNumRequests(4u);

    EXPECT_ANY_THROW(buildGraph().apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_gapi_age, m_gapi_gender),
                                        ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST_F(AgeGenderInferTest, ChangeOutputPrecision) {
    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        m_params.model_path, m_params.weights_path, m_params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgOutputPrecision(CV_8U);

    for (auto it : m_net.getOutputsInfo()) {
        it.second->setPrecision(IE::Precision::U8);
    }

    buildGraph().apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_gapi_age, m_gapi_gender),
                       ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(AgeGenderInferTest, ChangeSpecificOutputPrecison) {
    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        m_params.model_path, m_params.weights_path, m_params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgOutputPrecision({{"prob", CV_8U}});

    m_net.getOutputsInfo().at("prob")->setPrecision(IE::Precision::U8);

    buildGraph().apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_gapi_age, m_gapi_gender),
                       ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));
    validate();
}

TEST_F(AgeGenderInferTest, ThrowIfSetLayoutForImage) {
    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        m_params.model_path, m_params.weights_path, m_params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgOutputPrecision({{"prob", CV_8U}})
     .cfgInputLayout("NHWC");

    EXPECT_ANY_THROW(buildGraph().apply(ncvslideio::gin(m_in_mat), ncvslideio::gout(m_gapi_age, m_gapi_gender),
                                        ncvslideio::compile_args(ncvslideio::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, InferTensorWithPreproc) {
    initDLDTDataPath();

    ncvslideio::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    ncvslideio::Mat in_mat({1, 240, 320, 3}, CV_8U);
    ncvslideio::randu(in_mat, 0, 255);
    ncvslideio::Mat gapi_age, gapi_gender;

    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = ncvslideio::gimpl::ie::wrap::getPlugin(params);
        auto net           = ncvslideio::gimpl::ie::wrap::readNetwork(params);
        auto ii = net.getInputsInfo().at("data");

        ii->setPrecision(IE::Precision::U8);
        ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
        ii->setLayout(IE::Layout::NHWC);

        auto this_network  = ncvslideio::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        IE::TensorDesc desc{IE::Precision::U8, {1, 3, 240, 320}, IE::Layout::NHWC};
        auto blob =  IE::make_shared_blob<uint8_t>(desc, const_cast<uint8_t*>(in_mat.ptr<uint8_t>()));
        infer_request.SetBlob("data", blob);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
    G_API_NET(AgeGender, <AGInfo(ncvslideio::GMat)>, "test-age-gender");

    ncvslideio::GMat in;
    ncvslideio::GMat age, gender;
    std::tie(age, gender) = ncvslideio::gapi::infer<AgeGender>(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(age, gender));

    auto pp = ncvslideio::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgResize(ncvslideio::INTER_LINEAR)
     .cfgInputLayout("NHWC");

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(gapi_age, gapi_gender),
               ncvslideio::compile_args(ncvslideio::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(ncvslideio::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
