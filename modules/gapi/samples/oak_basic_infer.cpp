#include <algorithm>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/oak/infer.hpp>

const std::string keys =
    "{ h help              |             | Print this help message }"
    "{ detector            |             | Path to compiled .blob face detector model }"
    "{ duration            | 100         | Number of frames to pull from camera and run inference on }";

namespace custom {

G_API_NET(FaceDetector, <ncvslideio::GMat(ncvslideio::GFrame)>, "sample.custom.face-detector");

using GDetections = ncvslideio::GArray<ncvslideio::Rect>;
using GSize       = ncvslideio::GOpaque<ncvslideio::Size>;
using GPrims      = ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>;

G_API_OP(BBoxes, <GPrims(GDetections)>, "sample.custom.b-boxes") {
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GArrayDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVBBoxes, BBoxes) {
    // This kernel converts the rectangles into G-API's
    // rendering primitives
    static void run(const std::vector<ncvslideio::Rect> &in_face_rcs,
                          std::vector<ncvslideio::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const ncvslideio::Rect &rc, const ncvslideio::Scalar &clr) {
            return ncvslideio::gapi::wip::draw::Rect(rc, clr, 2);
        };
        for (auto &&rc : in_face_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0))); // green
        }
    }
};

} // namespace custom

int main(int argc, char *argv[]) {
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const auto det_name = cmd.get<std::string>("detector");
    const auto duration = cmd.get<int>("duration");

    if (det_name.empty()) {
        std::cerr << "FATAL: path to detection model is not provided for the sample."
                  << "Please specify it with --detector options."
                  << std::endl;
        return 1;
    }

    // Prepare G-API kernels and networks packages:
    auto detector = ncvslideio::gapi::oak::Params<custom::FaceDetector>(det_name);
    auto networks = ncvslideio::gapi::networks(detector);

    auto kernels = ncvslideio::gapi::combine(
        ncvslideio::gapi::kernels<custom::OCVBBoxes>(),
        ncvslideio::gapi::oak::kernels());

    auto args = ncvslideio::compile_args(kernels, networks);

    // Initialize graph structure
    ncvslideio::GFrame in;
    ncvslideio::GFrame copy = ncvslideio::gapi::oak::copy(in); // NV12 transfered to host + passthrough copy for infer
    ncvslideio::GOpaque<ncvslideio::Size> sz = ncvslideio::gapi::streaming::size(copy);

    // infer is not affected by the actual copy here
    ncvslideio::GMat blob = ncvslideio::gapi::infer<custom::FaceDetector>(copy);
    // FIXME: OAK infer detects faces slightly out of frame bounds
    ncvslideio::GArray<ncvslideio::Rect> rcs = ncvslideio::gapi::parseSSD(blob, sz, 0.5f, true, false);
    auto rendered = ncvslideio::gapi::wip::draw::renderFrame(copy, custom::BBoxes::on(rcs));
    // on-the-fly conversion NV12->BGR
    ncvslideio::GMat out = ncvslideio::gapi::streaming::BGR(rendered);

    auto pipeline  = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out, rcs))
        .compileStreaming(std::move(args));

    // Graph execution
    pipeline.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::oak::ColorCamera>());
    pipeline.start();

    ncvslideio::Mat out_mat;
    std::vector<ncvslideio::Rect> out_dets;
    int frames = 0;
    while (pipeline.pull(ncvslideio::gout(out_mat, out_dets))) {
        std::string name = "oak_infer_frame_" + std::to_string(frames) + ".png";

        ncvslideio::imwrite(name, out_mat);

        if (!out_dets.empty()) {
            std::cout << "Got " << out_dets.size() << " detections on frame #" << frames << std::endl;
        }

        ++frames;
        if (frames == duration) {
            pipeline.stop();
            break;
        }
    }
    std::cout << "Pipeline finished. Processed " << frames << " frames" << std::endl;
    return 0;
}
