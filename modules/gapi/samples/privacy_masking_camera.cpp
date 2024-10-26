#include <algorithm>
#include <iostream>
#include <cctype>

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

const std::string about =
    "This is an OpenCV-based version of Privacy Masking Camera example";
const std::string keys =
    "{ h help |                                                  | Print this help message }"
    "{ input  |                                                  | Path to the input video file }"
    "{ platm  | vehicle-license-plate-detection-barrier-0106.xml | Path to OpenVINO IE vehicle/plate detection model (.xml) }"
    "{ platd  | CPU                                              | Target device for vehicle/plate detection model (e.g. CPU, GPU, VPU, ...) }"
    "{ facem  | face-detection-retail-0005.xml                   | Path to OpenVINO IE face detection model (.xml) }"
    "{ faced  | CPU                                              | Target device for face detection model (e.g. CPU, GPU, VPU, ...) }"
    "{ trad   | false                                            | Run processing in a traditional (non-pipelined) way }"
    "{ noshow | false                                            | Don't display UI (improves performance) }";

namespace {

std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext = model_path.substr(sz - EXT_LEN);

    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return static_cast<unsigned char>(std::tolower(c)); });
    CV_Assert(ext == ".xml");

    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // namespace

namespace custom {

G_API_NET(VehLicDetector, <ncvslideio::GMat(ncvslideio::GMat)>, "vehicle-license-plate-detector");
G_API_NET(FaceDetector,   <ncvslideio::GMat(ncvslideio::GMat)>,                  "face-detector");

using GDetections = ncvslideio::GArray<ncvslideio::Rect>;

using GPrims = ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>;

G_API_OP(ToMosaic, <GPrims(GDetections, GDetections)>, "custom.privacy_masking.to_mosaic") {
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GArrayDesc &, const ncvslideio::GArrayDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(OCVToMosaic, ToMosaic) {
    static void run(const std::vector<ncvslideio::Rect> &in_plate_rcs,
                    const std::vector<ncvslideio::Rect> &in_face_rcs,
                          std::vector<ncvslideio::gapi::wip::draw::Prim> &out_prims) {
        out_prims.clear();
        const auto cvt = [](ncvslideio::Rect rc) {
            // Align the mosaic region to mosaic block size
            const int BLOCK_SIZE = 24;
            const int dw = BLOCK_SIZE - (rc.width  % BLOCK_SIZE);
            const int dh = BLOCK_SIZE - (rc.height % BLOCK_SIZE);
            rc.width  += dw;
            rc.height += dh;
            rc.x      -= dw / 2;
            rc.y      -= dh / 2;
            return ncvslideio::gapi::wip::draw::Mosaic{rc, BLOCK_SIZE, 0};
        };
        for (auto &&rc : in_plate_rcs) { out_prims.emplace_back(cvt(rc)); }
        for (auto &&rc : in_face_rcs)  { out_prims.emplace_back(cvt(rc)); }
    }
};

} // namespace custom

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const bool no_show = cmd.get<bool>("noshow");
    const bool run_trad = cmd.get<bool>("trad");

    ncvslideio::GMat in;
    ncvslideio::GMat blob_plates = ncvslideio::gapi::infer<custom::VehLicDetector>(in);
    ncvslideio::GMat blob_faces  = ncvslideio::gapi::infer<custom::FaceDetector>(in);
    // VehLicDetector from Open Model Zoo marks vehicles with label "1" and
    // license plates with label "2", filter out license plates only.
    ncvslideio::GOpaque<ncvslideio::Size> sz = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GArray<ncvslideio::Rect> rc_plates, rc_faces;
    ncvslideio::GArray<int> labels;
    std::tie(rc_plates, labels) = ncvslideio::gapi::parseSSD(blob_plates, sz, 0.5f, 2);
    // Face detector produces faces only so there's no need to filter by label,
    // pass "-1".
    std::tie(rc_faces, labels) = ncvslideio::gapi::parseSSD(blob_faces, sz, 0.5f, -1);
    ncvslideio::GMat out = ncvslideio::gapi::wip::draw::render3ch(in, custom::ToMosaic::on(rc_plates, rc_faces));
    ncvslideio::GComputation graph(in, out);

    const auto plate_model_path = cmd.get<std::string>("platm");
    auto plate_net = ncvslideio::gapi::ie::Params<custom::VehLicDetector> {
        plate_model_path,                // path to topology IR
        weights_path(plate_model_path),  // path to weights
        cmd.get<std::string>("platd"),   // device specifier
    };
    const auto face_model_path = cmd.get<std::string>("facem");
    auto face_net = ncvslideio::gapi::ie::Params<custom::FaceDetector> {
        face_model_path,                 // path to topology IR
        weights_path(face_model_path),   // path to weights
        cmd.get<std::string>("faced"),   // device specifier
    };
    auto kernels = ncvslideio::gapi::kernels<custom::OCVToMosaic>();
    auto networks = ncvslideio::gapi::networks(plate_net, face_net);

    ncvslideio::TickMeter tm;
    ncvslideio::Mat out_frame;
    std::size_t frames = 0u;
    std::cout << "Reading " << input << std::endl;

    if (run_trad) {
        ncvslideio::Mat in_frame;
        ncvslideio::VideoCapture cap(input);
        cap >> in_frame;

        auto exec = graph.compile(ncvslideio::descr_of(in_frame), ncvslideio::compile_args(kernels, networks));
        tm.start();
        do {
            exec(in_frame, out_frame);
            if (!no_show) {
                ncvslideio::imshow("Out", out_frame);
                ncvslideio::waitKey(1);
            }
            frames++;
        } while (cap.read(in_frame));
        tm.stop();
    } else {
        auto pipeline = graph.compileStreaming(ncvslideio::compile_args(kernels, networks));
        pipeline.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input));
        pipeline.start();
        tm.start();

        while (pipeline.pull(ncvslideio::gout(out_frame))) {
            frames++;
            if (!no_show) {
                ncvslideio::imshow("Out", out_frame);
                ncvslideio::waitKey(1);
            }
        }

        tm.stop();
    }

    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
