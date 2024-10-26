#include <chrono>
#include <iomanip>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/infer.hpp"
#include "opencv2/gapi/infer/ie.hpp"
#include "opencv2/gapi/infer/onnx.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace {
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ fdm    |   | IE face detection model IR }"
    "{ fdw    |   | IE face detection model weights }"
    "{ fdd    |   | IE face detection device }"
    "{ emom   |   | ONNX emotions recognition model }"
    "{ output |   | (Optional) Path to an output video file }"
    ;
} // namespace

namespace custom {
G_API_NET(Faces, <ncvslideio::GMat(ncvslideio::GMat)>, "face-detector");
G_API_NET(Emotions, <ncvslideio::GMat(ncvslideio::GMat)>, "emotions-recognition");

G_API_OP(PostProc, <ncvslideio::GArray<ncvslideio::Rect>(ncvslideio::GMat, ncvslideio::GMat)>, "custom.fd_postproc") {
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc &, const ncvslideio::GMatDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const ncvslideio::Mat &in_ssd_result,
                    const ncvslideio::Mat &in_frame,
                    std::vector<ncvslideio::Rect> &out_faces) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const ncvslideio::Size upscale = in_frame.size();
        const ncvslideio::Rect surface({0,0}, upscale);

        out_faces.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) {
                continue;
            }

            ncvslideio::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            out_faces.push_back(rc & surface);
        }
    }
};
//! [Postproc]

} // namespace custom

namespace labels {
// Labels as defined in
// https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
//
const std::string emotions[] = {
    "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"
};
namespace {
template<typename Iter>
std::vector<float> softmax(Iter begin, Iter end) {
    std::vector<float> prob(end - begin, 0.f);
    std::transform(begin, end, prob.begin(), [](float x) { return std::exp(x); });
    float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
    for (int i = 0; i < static_cast<int>(prob.size()); i++)
        prob[i] /= sum;
    return prob;
}

void DrawResults(ncvslideio::Mat &frame,
                 const std::vector<ncvslideio::Rect> &faces,
                 const std::vector<ncvslideio::Mat>  &out_emotions) {
    CV_Assert(faces.size() == out_emotions.size());

    for (auto it = faces.begin(); it != faces.end(); ++it) {
        const auto idx = std::distance(faces.begin(), it);
        const auto &rc = *it;

        const float *emotions_data = out_emotions[idx].ptr<float>();
        auto sm = softmax(emotions_data, emotions_data + 8);
        const auto emo_id = std::max_element(sm.begin(), sm.end()) - sm.begin();

        const int ATTRIB_OFFSET = 15;
        ncvslideio::rectangle(frame, rc, {0, 255, 0},  4);
        ncvslideio::putText(frame, emotions[emo_id],
                    ncvslideio::Point(rc.x, rc.y - ATTRIB_OFFSET),
                    ncvslideio::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    ncvslideio::Scalar(0, 0, 255));

        std::cout << emotions[emo_id] << " at " << rc << std::endl;
    }
}
} // anonymous namespace
} // namespace labels

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");

    // OpenVINO FD parameters here
    auto det_net = ncvslideio::gapi::ie::Params<custom::Faces> {
        cmd.get<std::string>("fdm"),   // read cmd args: path to topology IR
        cmd.get<std::string>("fdw"),   // read cmd args: path to weights
        cmd.get<std::string>("fdd"),   // read cmd args: device specifier
    };

    // ONNX Emotions parameters here
    auto emo_net = ncvslideio::gapi::onnx::Params<custom::Emotions> {
        cmd.get<std::string>("emom"),   // read cmd args: path to the ONNX model
    }.cfgNormalize({false}); // model accepts 0..255 range in FP32

    auto kernels = ncvslideio::gapi::kernels<custom::OCVPostProc>();
    auto networks = ncvslideio::gapi::networks(det_net, emo_net);

    ncvslideio::GMat in;
    ncvslideio::GMat bgr = ncvslideio::gapi::copy(in);
    ncvslideio::GMat frame = ncvslideio::gapi::streaming::desync(bgr);
    ncvslideio::GMat detections = ncvslideio::gapi::infer<custom::Faces>(frame);
    ncvslideio::GArray<ncvslideio::Rect> faces = custom::PostProc::on(detections, frame);
    ncvslideio::GArray<ncvslideio::GMat> emotions = ncvslideio::gapi::infer<custom::Emotions>(faces, frame);
    auto pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(bgr, faces, emotions))
        .compileStreaming(ncvslideio::compile_args(kernels, networks));

    auto in_src = ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input);
    pipeline.setSource(ncvslideio::gin(in_src));

    ncvslideio::util::optional<ncvslideio::Mat>               out_frame;
    ncvslideio::util::optional<std::vector<ncvslideio::Rect>> out_faces;
    ncvslideio::util::optional<std::vector<ncvslideio::Mat>>  out_emotions;

    ncvslideio::Mat               last_mat;
    std::vector<ncvslideio::Rect> last_faces;
    std::vector<ncvslideio::Mat>  last_emotions;

    ncvslideio::VideoWriter writer;
    ncvslideio::TickMeter tm;
    std::size_t frames = 0u;

    tm.start();
    pipeline.start();
    while (pipeline.pull(ncvslideio::gout(out_frame, out_faces, out_emotions))) {
        ++frames;
        if (out_faces && out_emotions) {
            last_faces = *out_faces;
            last_emotions = *out_emotions;
        }
        if (out_frame) {
            last_mat = *out_frame;
            labels::DrawResults(last_mat, last_faces, last_emotions);

            if (!output.empty()) {
                if (!writer.isOpened()) {
                    const auto sz = ncvslideio::Size{last_mat.cols, last_mat.rows};
                    writer.open(output, ncvslideio::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
                    CV_Assert(writer.isOpened());
                }
                writer << last_mat;
            }
        }
        if (!last_mat.empty()) {
            ncvslideio::imshow("Out", last_mat);
            ncvslideio::waitKey(1);
        }
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
