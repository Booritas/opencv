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
#include <opencv2/gapi/infer/onnx.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/infer/parsers.hpp>

namespace custom {

G_API_NET(ObjDetector,   <ncvslideio::GMat(ncvslideio::GMat)>, "object-detector");

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
    static void run(const std::vector<ncvslideio::Rect> &in_obj_rcs,
                          std::vector<ncvslideio::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](const ncvslideio::Rect &rc, const ncvslideio::Scalar &clr) {
            return ncvslideio::gapi::wip::draw::Rect(rc, clr, 2);
        };
        for (auto &&rc : in_obj_rcs) {
            out_prims.emplace_back(cvt(rc, CV_RGB(0,255,0)));   // green
        }

        std::cout << "Detections:";
        for (auto &&rc : in_obj_rcs) std::cout << ' ' << rc;
        std::cout << std::endl;
    }
};

} // namespace custom

namespace {
void remap_ssd_ports(const std::unordered_map<std::string, ncvslideio::Mat> &onnx,
                           std::unordered_map<std::string, ncvslideio::Mat> &gapi) {
    // Assemble ONNX-processed outputs back to a single 1x1x200x7 blob
    // to preserve compatibility with OpenVINO-based SSD pipeline
    const ncvslideio::Mat &num_detections = onnx.at("num_detections:0");
    const ncvslideio::Mat &detection_boxes = onnx.at("detection_boxes:0");
    const ncvslideio::Mat &detection_scores = onnx.at("detection_scores:0");
    const ncvslideio::Mat &detection_classes = onnx.at("detection_classes:0");

    GAPI_Assert(num_detections.depth() == CV_32F);
    GAPI_Assert(detection_boxes.depth() == CV_32F);
    GAPI_Assert(detection_scores.depth() == CV_32F);
    GAPI_Assert(detection_classes.depth() == CV_32F);

    ncvslideio::Mat &ssd_output = gapi.at("detection_output");

    const int num_objects = static_cast<int>(num_detections.ptr<float>()[0]);
    const float *in_boxes = detection_boxes.ptr<float>();
    const float *in_scores = detection_scores.ptr<float>();
    const float *in_classes = detection_classes.ptr<float>();
    float *ptr = ssd_output.ptr<float>();

    for (int i = 0; i < num_objects; i++) {
        ptr[0] = 0.f;               // "image_id"
        ptr[1] = in_classes[i];     // "label"
        ptr[2] = in_scores[i];      // "confidence"
        ptr[3] = in_boxes[4*i + 1]; // left
        ptr[4] = in_boxes[4*i + 0]; // top
        ptr[5] = in_boxes[4*i + 3]; // right
        ptr[6] = in_boxes[4*i + 2]; // bottom

        ptr      += 7;
        in_boxes += 4;
    }
    if (num_objects < ssd_output.size[2]-1) {
        // put a -1 mark at the end of output blob if there is space left
        ptr[0] = -1.f;
    }
}
} // anonymous namespace

const std::string keys =
    "{ h help | | Print this help message }"
    "{ input  | | Path to the input video file }"
    "{ output | | (Optional) path to output video file }"
    "{ detm   | | Path to an ONNX SSD object detection model (.onnx) }"
    ;

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto obj_model_path = cmd.get<std::string>("detm");

    auto obj_net = ncvslideio::gapi::onnx::Params<custom::ObjDetector>{obj_model_path}
        .cfgOutputLayers({"detection_output"})
        .cfgPostProc({ncvslideio::GMatDesc{CV_32F, {1,1,200,7}}}, remap_ssd_ports);
    auto kernels = ncvslideio::gapi::kernels<custom::OCVBBoxes>();
    auto networks = ncvslideio::gapi::networks(obj_net);

    // Now build the graph
    ncvslideio::GMat in;
    auto blob = ncvslideio::gapi::infer<custom::ObjDetector>(in);
    ncvslideio::GArray<ncvslideio::Rect> rcs =
        ncvslideio::gapi::parseSSD(blob, ncvslideio::gapi::streaming::size(in), 0.5f, true, true);
    auto  out = ncvslideio::gapi::wip::draw::render3ch(in, custom::BBoxes::on(rcs));
    ncvslideio::GStreamingCompiled pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
        .compileStreaming(ncvslideio::compile_args(kernels, networks));

    auto inputs = ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input));

    // The execution part
    pipeline.setSource(std::move(inputs));

    ncvslideio::TickMeter tm;
    ncvslideio::VideoWriter writer;
    size_t frames = 0u;
    ncvslideio::Mat outMat;

    tm.start();
    pipeline.start();
    while (pipeline.pull(ncvslideio::gout(outMat))) {
        ++frames;
        ncvslideio::imshow("Out", outMat);
        ncvslideio::waitKey(1);
        if (!output.empty()) {
            if (!writer.isOpened()) {
                const auto sz = ncvslideio::Size{outMat.cols, outMat.rows};
                writer.open(output, ncvslideio::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
                CV_Assert(writer.isOpened());
            }
            writer << outMat;
        }
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
