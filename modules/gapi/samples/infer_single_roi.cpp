#include <algorithm>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/infer/parsers.hpp>

const std::string keys =
    "{ h help |                              | Print this help message }"
    "{ input  |                              | Path to the input video file }"
    "{ facem  | face-detection-adas-0001.xml | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced  | CPU                          | Target device for face detection model (e.g. CPU, GPU, VPU, ...) }"
    "{ r roi  | -1,-1,-1,-1                  | Region of interest (ROI) to use for inference. Identified automatically when not set }";

namespace {

std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){
            return static_cast<unsigned char>(std::tolower(c));
        });
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}

ncvslideio::util::optional<ncvslideio::Rect> parse_roi(const std::string &rc) {
    ncvslideio::Rect rv;
    char delim[3];

    std::stringstream is(rc);
    is >> rv.x >> delim[0] >> rv.y >> delim[1] >> rv.width >> delim[2] >> rv.height;
    if (is.bad()) {
        return ncvslideio::util::optional<ncvslideio::Rect>(); // empty value
    }
    const auto is_delim = [](char c) {
        return c == ',';
    };
    if (!std::all_of(std::begin(delim), std::end(delim), is_delim)) {
        return ncvslideio::util::optional<ncvslideio::Rect>(); // empty value

    }
    if (rv.x < 0 || rv.y < 0 || rv.width <= 0 || rv.height <= 0) {
        return ncvslideio::util::optional<ncvslideio::Rect>(); // empty value
    }
    return ncvslideio::util::make_optional(std::move(rv));
}

} // namespace

namespace custom {

G_API_NET(FaceDetector,   <ncvslideio::GMat(ncvslideio::GMat)>, "face-detector");

using GDetections = ncvslideio::GArray<ncvslideio::Rect>;
using GRect       = ncvslideio::GOpaque<ncvslideio::Rect>;
using GSize       = ncvslideio::GOpaque<ncvslideio::Size>;
using GPrims      = ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>;

G_API_OP(LocateROI, <GRect(ncvslideio::GMat)>, "sample.custom.locate-roi") {
    static ncvslideio::GOpaqueDesc outMeta(const ncvslideio::GMatDesc &) {
        return ncvslideio::empty_gopaque_desc();
    }
};

G_API_OP(BBoxes, <GPrims(GDetections, GRect)>, "sample.custom.b-boxes") {
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GArrayDesc &, const ncvslideio::GOpaqueDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVLocateROI, LocateROI) {
    // This is the place where we can run extra analytics
    // on the input image frame and select the ROI (region
    // of interest) where we want to detect our objects (or
    // run any other inference).
    //
    // Currently it doesn't do anything intelligent,
    // but only crops the input image to square (this is
    // the most convenient aspect ratio for detectors to use)

    static void run(const ncvslideio::Mat &in_mat, ncvslideio::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        const auto center = ncvslideio::Point{in_mat.cols/2, in_mat.rows/2};
        auto sqside = std::min(in_mat.cols, in_mat.rows);

        // Now build the central square ROI
        out_rect = ncvslideio::Rect{ center.x - sqside/2
                           , center.y - sqside/2
                           , sqside
                           , sqside
                           };
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<ncvslideio::Rect> &in_face_rcs,
                    const             ncvslideio::Rect  &in_roi,
                          std::vector<ncvslideio::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const ncvslideio::Rect &rc, const ncvslideio::Scalar &clr) {
            return ncvslideio::gapi::wip::draw::Rect(rc, clr, 2);
        };
        out_prims.emplace_back(cvt(in_roi, CV_RGB(0,255,255))); // cyan
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }
    }
};

} // namespace custom

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input = cmd.get<std::string>("input");
    const auto opt_roi = parse_roi(cmd.get<std::string>("roi"));

    const auto face_model_path = cmd.get<std::string>("facem");
    auto face_net = ncvslideio::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        weights_path(face_model_path),   // path to weights
        cmd.get<std::string>("faced"),   // device specifier
    };
    auto kernels = ncvslideio::gapi::kernels
        <custom::OCVLocateROI
        , custom::OCVBBoxes>();
    auto networks = ncvslideio::gapi::networks(face_net);

    // Now build the graph. The graph structure may vary
    // passed on the input parameters
    ncvslideio::GStreamingCompiled pipeline;
    auto inputs = ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input));

    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> sz = ncvslideio::gapi::streaming::size(in);
    if (opt_roi.has_value()) {
        // Use the value provided by user
        std::cout << "Will run inference for static region "
                  << opt_roi.value()
                  << " only"
                  << std::endl;
        ncvslideio::GOpaque<ncvslideio::Rect> in_roi;
        auto blob = ncvslideio::gapi::infer<custom::FaceDetector>(in_roi, in);
        ncvslideio::GArray<ncvslideio::Rect> rcs = ncvslideio::gapi::parseSSD(blob, sz, 0.5f, true, true);
        auto  out = ncvslideio::gapi::wip::draw::render3ch(in, custom::BBoxes::on(rcs, in_roi));
        pipeline  = ncvslideio::GComputation(ncvslideio::GIn(in, in_roi), ncvslideio::GOut(out))
            .compileStreaming(ncvslideio::compile_args(kernels, networks));

        // Since the ROI to detect is manual, make it part of the input vector
        inputs.push_back(ncvslideio::gin(opt_roi.value())[0]);
    } else {
        // Automatically detect ROI to infer. Make it output parameter
        std::cout << "ROI is not set or invalid. Locating it automatically"
                  << std::endl;
        ncvslideio::GOpaque<ncvslideio::Rect> roi = custom::LocateROI::on(in);
        auto blob = ncvslideio::gapi::infer<custom::FaceDetector>(roi, in);
        ncvslideio::GArray<ncvslideio::Rect> rcs = ncvslideio::gapi::parseSSD(blob, sz, 0.5f, true, true);
        auto  out = ncvslideio::gapi::wip::draw::render3ch(in, custom::BBoxes::on(rcs, roi));
        pipeline  = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
            .compileStreaming(ncvslideio::compile_args(kernels, networks));
    }

    // The execution part
    pipeline.setSource(std::move(inputs));
    pipeline.start();

    ncvslideio::Mat out;
    size_t frames = 0u;
    ncvslideio::TickMeter tm;
    tm.start();
    while (pipeline.pull(ncvslideio::gout(out))) {
        ncvslideio::imshow("Out", out);
        ncvslideio::waitKey(1);
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
