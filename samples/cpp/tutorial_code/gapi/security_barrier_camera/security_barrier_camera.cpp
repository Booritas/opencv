#include "opencv2/opencv_modules.hpp"
#include <iostream>
#if defined(HAVE_OPENCV_GAPI)

#include <chrono>
#include <iomanip>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/gapi/infer.hpp"
#include "opencv2/gapi/infer/ie.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "opencv2/gapi/streaming/cap.hpp"
#include "opencv2/highgui.hpp"

const std::string about =
    "This is an OpenCV-based version of Security Barrier Camera example";
const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ detm   |   | IE vehicle/license plate detection model IR }"
    "{ detw   |   | IE vehicle/license plate detection model weights }"
    "{ detd   |   | IE vehicle/license plate detection model device }"
    "{ vehm   |   | IE vehicle attributes model IR }"
    "{ vehw   |   | IE vehicle attributes model weights }"
    "{ vehd   |   | IE vehicle attributes model device }"
    "{ lprm   |   | IE license plate recognition model IR }"
    "{ lprw   |   | IE license plate recognition model weights }"
    "{ lprd   |   | IE license plate recognition model device }"
    "{ pure   |   | When set, no output is displayed. Useful for benchmarking }"
    "{ ser    |   | When set, runs a regular (serial) pipeline }";

namespace {
struct Avg {
    struct Elapsed {
        explicit Elapsed(double ms) : ss(ms/1000.), mm(static_cast<int>(ss)/60) {}
        const double ss;
        const int    mm;
    };

    using MS = std::chrono::duration<double, std::ratio<1, 1000>>;
    using TS = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TS started;

    void    start() { started = now(); }
    TS      now() const { return std::chrono::high_resolution_clock::now(); }
    double  tick() const { return std::chrono::duration_cast<MS>(now() - started).count(); }
    Elapsed elapsed() const { return Elapsed{tick()}; }
    double  fps(std::size_t n) const { return static_cast<double>(n) / (tick() / 1000.); }
};
std::ostream& operator<<(std::ostream &os, const Avg::Elapsed &e) {
    os << e.mm << ':' << (e.ss - 60*e.mm);
    return os;
}
} // namespace


namespace custom {
G_API_NET(VehicleLicenseDetector, <ncvslideio::GMat(ncvslideio::GMat)>, "vehicle-license-plate-detector");

using Attrs = std::tuple<ncvslideio::GMat, ncvslideio::GMat>;
G_API_NET(VehicleAttributes,      <Attrs(ncvslideio::GMat)>,    "vehicle-attributes");
G_API_NET(LPR,                    <ncvslideio::GMat(ncvslideio::GMat)>, "license-plate-recognition");

using GVehiclesPlates = std::tuple< ncvslideio::GArray<ncvslideio::Rect>
                                  , ncvslideio::GArray<ncvslideio::Rect> >;
G_API_OP_M(ProcessDetections,
           <GVehiclesPlates(ncvslideio::GMat, ncvslideio::GMat)>,
           "custom.security_barrier.detector.postproc") {
    static std::tuple<ncvslideio::GArrayDesc,ncvslideio::GArrayDesc>
    outMeta(const ncvslideio::GMatDesc &, const ncvslideio::GMatDesc) {
        // FIXME: Need to get rid of this - literally there's nothing useful
        return std::make_tuple(ncvslideio::empty_array_desc(), ncvslideio::empty_array_desc());
    }
};

GAPI_OCV_KERNEL(OCVProcessDetections, ProcessDetections) {
    static void run(const ncvslideio::Mat &in_ssd_result,
                    const ncvslideio::Mat &in_frame,
                    std::vector<ncvslideio::Rect> &out_vehicles,
                    std::vector<ncvslideio::Rect> &out_plates) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const ncvslideio::Size upscale = in_frame.size();
        const ncvslideio::Rect surface({0,0}, upscale);

        out_vehicles.clear();
        out_plates.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float label      = data[i * OBJECT_SIZE + 1];
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];

            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) { // fixme: hard-coded snapshot
                continue;
            }

            ncvslideio::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;

            using PT = ncvslideio::Point;
            using SZ = ncvslideio::Size;
            switch (static_cast<int>(label)) {
            case 1: out_vehicles.push_back(rc & surface); break;
            case 2: out_plates.emplace_back((rc-PT(15,15)+SZ(30,30)) & surface); break;
            default: CV_Assert(false && "Unknown object class");
            }
        }
    }
};
} // namespace custom

namespace labels {
const std::string colors[] = {
    "white", "gray", "yellow", "red", "green", "blue", "black"
};
const std::string types[] = {
    "car", "van", "truck", "bus"
};
const std::vector<std::string> license_text = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
    "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
    "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
    "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
    "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
    "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
    "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
    "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
    "<Zhejiang>", "<police>",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
};
namespace {
void DrawResults(ncvslideio::Mat &frame,
                 const std::vector<ncvslideio::Rect> &vehicles,
                 const std::vector<ncvslideio::Mat>  &out_colors,
                 const std::vector<ncvslideio::Mat>  &out_types,
                 const std::vector<ncvslideio::Rect> &plates,
                 const std::vector<ncvslideio::Mat>  &out_numbers) {
    CV_Assert(vehicles.size() == out_colors.size());
    CV_Assert(vehicles.size() == out_types.size());
    CV_Assert(plates.size()   == out_numbers.size());

    for (auto it = vehicles.begin(); it != vehicles.end(); ++it) {
        const auto idx = std::distance(vehicles.begin(), it);
        const auto &rc = *it;

        const float *colors_data = out_colors[idx].ptr<float>();
        const float *types_data  = out_types [idx].ptr<float>();
        const auto color_id = std::max_element(colors_data, colors_data + 7) - colors_data;
        const auto  type_id = std::max_element(types_data,  types_data  + 4) - types_data;

        const int ATTRIB_OFFSET = 25;
        ncvslideio::rectangle(frame, rc, {0, 255, 0},  4);
        ncvslideio::putText(frame, labels::colors[color_id],
                    ncvslideio::Point(rc.x + 5, rc.y + ATTRIB_OFFSET),
                    ncvslideio::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    ncvslideio::Scalar(255, 0, 0));
        ncvslideio::putText(frame, labels::types[type_id],
                    ncvslideio::Point(rc.x + 5, rc.y + ATTRIB_OFFSET * 2),
                    ncvslideio::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    ncvslideio::Scalar(255, 0, 0));
    }

    for (auto it = plates.begin(); it != plates.end(); ++it) {
        const int MAX_LICENSE = 88;
        const int LPR_OFFSET  = 50;

        const auto &rc   = *it;
        const auto idx   = std::distance(plates.begin(), it);

        std::string result;
        const auto *lpr_data = out_numbers[idx].ptr<float>();
        for (int i = 0; i < MAX_LICENSE; i++) {
            if (lpr_data[i] == -1) break;
            result += labels::license_text[static_cast<size_t>(lpr_data[i])];
        }

        const int y_pos = std::max(0, rc.y + rc.height - LPR_OFFSET);
        ncvslideio::rectangle(frame, rc, {0, 0, 255},  4);
        ncvslideio::putText(frame, result,
                    ncvslideio::Point(rc.x, y_pos),
                    ncvslideio::FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    ncvslideio::Scalar(0, 0, 255));
    }
}

void DrawFPS(ncvslideio::Mat &frame, std::size_t n, double fps) {
    std::ostringstream out;
    out << "FRAME " << n << ": "
        << std::fixed << std::setprecision(2) << fps
        << " FPS (AVG)";
    ncvslideio::putText(frame, out.str(),
                ncvslideio::Point(0, frame.rows),
                ncvslideio::FONT_HERSHEY_SIMPLEX,
                1,
                ncvslideio::Scalar(0, 0, 0),
                2);
}
} // anonymous namespace
} // namespace labels

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    const std::string input = cmd.get<std::string>("input");
    const bool no_show = cmd.get<bool>("pure");

    ncvslideio::GComputation pp([]() {
            ncvslideio::GMat in;
            ncvslideio::GMat detections          = ncvslideio::gapi::infer<custom::VehicleLicenseDetector>(in);
            ncvslideio::GArray<ncvslideio::Rect> vehicles;
            ncvslideio::GArray<ncvslideio::Rect> plates;
            std::tie(vehicles, plates)   = custom::ProcessDetections::on(detections, in);
            ncvslideio::GArray<ncvslideio::GMat> colors;
            ncvslideio::GArray<ncvslideio::GMat> types;
            std::tie(colors, types)      = ncvslideio::gapi::infer<custom::VehicleAttributes>(vehicles, in);
            ncvslideio::GArray<ncvslideio::GMat> numbers = ncvslideio::gapi::infer<custom::LPR>(plates, in);
            ncvslideio::GMat frame = ncvslideio::gapi::copy(in); // pass-through the input frame
            return ncvslideio::GComputation(ncvslideio::GIn(in),
                                    ncvslideio::GOut(frame, vehicles, colors, types, plates, numbers));
        });

    // Note: it might be very useful to have dimensions loaded at this point!
    auto det_net = ncvslideio::gapi::ie::Params<custom::VehicleLicenseDetector> {
        cmd.get<std::string>("detm"),   // path to topology IR
        cmd.get<std::string>("detw"),   // path to weights
        cmd.get<std::string>("detd"),   // device specifier
    };

    auto attr_net = ncvslideio::gapi::ie::Params<custom::VehicleAttributes> {
        cmd.get<std::string>("vehm"),   // path to topology IR
        cmd.get<std::string>("vehw"),   // path to weights
        cmd.get<std::string>("vehd"),   // device specifier
    }.cfgOutputLayers({ "color", "type" });

    // Fill a special LPR input (seq_ind) with a predefined value
    // First element is 0.f, the rest 87 are 1.f
    const std::vector<int> lpr_seq_dims = {88,1};
    ncvslideio::Mat lpr_seq(lpr_seq_dims, CV_32F, ncvslideio::Scalar(1.f));
    lpr_seq.ptr<float>()[0] = 0.f;
    auto lpr_net = ncvslideio::gapi::ie::Params<custom::LPR> {
        cmd.get<std::string>("lprm"),   // path to topology IR
        cmd.get<std::string>("lprw"),   // path to weights
        cmd.get<std::string>("lprd"),   // device specifier
    }.constInput("seq_ind", lpr_seq);

    auto kernels = ncvslideio::gapi::kernels<custom::OCVProcessDetections>();
    auto networks = ncvslideio::gapi::networks(det_net, attr_net, lpr_net);

    Avg avg;
    ncvslideio::Mat frame;
    std::vector<ncvslideio::Rect> vehicles, plates;
    std::vector<ncvslideio::Mat> out_colors;
    std::vector<ncvslideio::Mat> out_types;
    std::vector<ncvslideio::Mat> out_numbers;
    std::size_t frames = 0u;

    std::cout << "Reading " << input << std::endl;

    if (cmd.get<bool>("ser")) {
        std::cout << "Going serial..." << std::endl;
        ncvslideio::VideoCapture cap(input);

        auto cc = pp.compile(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size(1920,1080)},
                             ncvslideio::compile_args(kernels, networks));

        avg.start();
        while (ncvslideio::waitKey(1) < 0) {
            cap >> frame;
            if (frame.empty()) break;

            cc(ncvslideio::gin(frame),
               ncvslideio::gout(frame, vehicles, out_colors, out_types, plates, out_numbers));
            frames++;
            labels::DrawResults(frame, vehicles, out_colors, out_types, plates, out_numbers);
            labels::DrawFPS(frame, frames, avg.fps(frames));
            if (!no_show) ncvslideio::imshow("Out", frame);
        }
    } else {
        std::cout << "Going pipelined..." << std::endl;

        auto cc = pp.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size(1920,1080)},
                                      ncvslideio::compile_args(kernels, networks));

        cc.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input));

        avg.start();
        cc.start();

        // Implement different execution policies depending on the display option
        // for the best performance.
        while (cc.running()) {
            auto out_vector = ncvslideio::gout(frame, vehicles, out_colors, out_types, plates, out_numbers);
            if (no_show) {
                // This is purely a video processing. No need to balance with UI rendering.
                // Use a blocking pull() to obtain data. Break the loop if the stream is over.
                if (!cc.pull(std::move(out_vector)))
                    break;
            } else if (!cc.try_pull(std::move(out_vector))) {
                // Use a non-blocking try_pull() to obtain data.
                // If there's no data, let UI refresh (and handle keypress)
                if (ncvslideio::waitKey(1) >= 0) break;
                else continue;
            }
            // At this point we have data for sure (obtained in either blocking or non-blocking way).
            frames++;
            labels::DrawResults(frame, vehicles, out_colors, out_types, plates, out_numbers);
            labels::DrawFPS(frame, frames, avg.fps(frames));
            if (!no_show) ncvslideio::imshow("Out", frame);
        }
        cc.stop();
    }
    std::cout << "Processed " << frames << " frames in " << avg.elapsed() << std::endl;

    return 0;
}
#else
int main()
{
    std::cerr << "This tutorial code requires G-API module "
                 "with Inference Engine backend to run"
              << std::endl;
    return 1;
}
#endif  // HAVE_OPECV_GAPI
