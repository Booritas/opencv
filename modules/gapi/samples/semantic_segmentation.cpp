#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/operators.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#include <iomanip>

const std::string keys =
    "{ h help |                                     | Print this help message }"
    "{ desync | false                               | Desynchronize inference }"
    "{ input  |                                     | Path to the input video file }"
    "{ output |                                     | Path to the output video file }"
    "{ ssm    | semantic-segmentation-adas-0001.xml | Path to OpenVINO IE semantic segmentation model (.xml) }";

// 20 colors for 20 classes of semantic-segmentation-adas-0001
static std::vector<ncvslideio::Vec3b> colors = {
    { 0, 0, 0 },
    { 0, 0, 128 },
    { 0, 128, 0 },
    { 0, 128, 128 },
    { 128, 0, 0 },
    { 128, 0, 128 },
    { 128, 128, 0 },
    { 128, 128, 128 },
    { 0, 0, 64 },
    { 0, 0, 192 },
    { 0, 128, 64 },
    { 0, 128, 192 },
    { 128, 0, 64 },
    { 128, 0, 192 },
    { 128, 128, 64 },
    { 128, 128, 192 },
    { 0, 64, 0 },
    { 0, 64, 128 },
    { 0, 192, 0 },
    { 0, 192, 128 },
    { 128, 64, 0 }
};

namespace {
std::string get_weights_path(const std::string &model_path) {
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

bool isNumber(const std::string &str) {
    return !str.empty() && std::all_of(str.begin(), str.end(),
            [](unsigned char ch) { return std::isdigit(ch); });
}

std::string toStr(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << value;
    return ss.str();
}

void classesToColors(const ncvslideio::Mat &out_blob,
                           ncvslideio::Mat &mask_img) {
    const int H = out_blob.size[0];
    const int W = out_blob.size[1];

    mask_img.create(H, W, CV_8UC3);
    GAPI_Assert(out_blob.type() == CV_8UC1);
    const uint8_t* const classes = out_blob.ptr<uint8_t>();

    for (int rowId = 0; rowId < H; ++rowId) {
        for (int colId = 0; colId < W; ++colId) {
            uint8_t class_id = classes[rowId * W + colId];
            mask_img.at<ncvslideio::Vec3b>(rowId, colId) =
                class_id < colors.size()
                ? colors[class_id]
                : ncvslideio::Vec3b{0, 0, 0}; // NB: sample supports 20 classes
        }
    }
}

void probsToClasses(const ncvslideio::Mat& probs, ncvslideio::Mat& classes) {
     const int C = probs.size[1];
     const int H = probs.size[2];
     const int W = probs.size[3];

     classes.create(H, W, CV_8UC1);
     GAPI_Assert(probs.depth() == CV_32F);
     float* out_p       = reinterpret_cast<float*>(probs.data);
     uint8_t* classes_p = reinterpret_cast<uint8_t*>(classes.data);

     for (int h = 0; h < H; ++h) {
         for (int w = 0; w < W; ++w) {
             double max = 0;
             int class_id = 0;
             for (int c = 0; c < C; ++c) {
                int idx = c * H * W + h * W + w;
                    if (out_p[idx] > max) {
                        max = out_p[idx];
                        class_id = c;
                    }
             }
             classes_p[h * W + w] = static_cast<uint8_t>(class_id);
         }
     }
}

} // anonymous namespace

namespace vis {

static void putText(ncvslideio::Mat& mat, const ncvslideio::Point &position, const std::string &message) {
    auto fontFace = ncvslideio::FONT_HERSHEY_COMPLEX;
    int thickness = 2;
    ncvslideio::Scalar color = {200, 10, 10};
    double fontScale = 0.65;

    ncvslideio::putText(mat, message, position, fontFace,
                fontScale, ncvslideio::Scalar(255, 255, 255), thickness + 1);
    ncvslideio::putText(mat, message, position, fontFace, fontScale, color, thickness);
}

static void drawResults(ncvslideio::Mat &img, const ncvslideio::Mat &color_mask) {
    img = img / 2 + color_mask / 2;
}

} // namespace vis

namespace custom {
G_API_OP(PostProcessing, <ncvslideio::GMat(ncvslideio::GMat, ncvslideio::GMat)>, "sample.custom.post_processing") {
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in, const ncvslideio::GMatDesc &) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVPostProcessing, PostProcessing) {
    static void run(const ncvslideio::Mat &in, const ncvslideio::Mat &out_blob, ncvslideio::Mat &out) {
        int C = -1, H = -1, W = -1;
        if (out_blob.size.dims() == 4u) {
            C = 1; H = 2, W = 3;
        } else if (out_blob.size.dims() == 3u) {
            C = 0; H = 1, W = 2;
        } else {
            throw std::logic_error(
                    "Number of dimmensions for model output must be 3 or 4!");
        }
        ncvslideio::Mat classes;
        // NB: If output has more than single plane, it contains probabilities
        // otherwise class id.
        if (out_blob.size[C] > 1) {
            probsToClasses(out_blob, classes);
        } else {
            if (out_blob.depth() != CV_32S) {
                throw std::logic_error(
                        "Single channel output must have integer precision!");
            }
            ncvslideio::Mat view(out_blob.size[H], // cols
                         out_blob.size[W], // rows
                         CV_32SC1,
                         out_blob.data);
            view.convertTo(classes, CV_8UC1);
        }
        ncvslideio::Mat mask_img;
        classesToColors(classes, mask_img);
        ncvslideio::resize(mask_img, out, in.size(), 0, 0, ncvslideio::INTER_NEAREST);
    }
};
} // namespace custom

int main(int argc, char *argv[]) {
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Prepare parameters first
    const std::string input  = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const auto model_path    = cmd.get<std::string>("ssm");
    const bool desync        = cmd.get<bool>("desync");
    const auto weights_path  = get_weights_path(model_path);
    const auto device        = "CPU";
    G_API_NET(SemSegmNet, <ncvslideio::GMat(ncvslideio::GMat)>, "semantic-segmentation");
    const auto net = ncvslideio::gapi::ie::Params<SemSegmNet> {
        model_path, weights_path, device
    };
    const auto kernels = ncvslideio::gapi::kernels<custom::OCVPostProcessing>();
    const auto networks = ncvslideio::gapi::networks(net);

    // Now build the graph
    ncvslideio::GMat in;
    ncvslideio::GMat bgr = ncvslideio::gapi::copy(in);
    ncvslideio::GMat frame = desync ? ncvslideio::gapi::streaming::desync(bgr) : bgr;
    ncvslideio::GMat out_blob = ncvslideio::gapi::infer<SemSegmNet>(frame);
    ncvslideio::GMat out = custom::PostProcessing::on(frame, out_blob);

    ncvslideio::GStreamingCompiled pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(bgr, out))
        .compileStreaming(ncvslideio::compile_args(kernels, networks,
                          ncvslideio::gapi::streaming::queue_capacity{1}));

    std::shared_ptr<ncvslideio::gapi::wip::GCaptureSource> source;
    if (isNumber(input)) {
        source = std::make_shared<ncvslideio::gapi::wip::GCaptureSource>(
            std::stoi(input),
            std::map<int, double> {
              {ncvslideio::CAP_PROP_FRAME_WIDTH, 1280},
              {ncvslideio::CAP_PROP_FRAME_HEIGHT, 720},
              {ncvslideio::CAP_PROP_BUFFERSIZE, 1},
              {ncvslideio::CAP_PROP_AUTOFOCUS, true}
            }
        );
    } else {
        source = std::make_shared<ncvslideio::gapi::wip::GCaptureSource>(input);
    }
    auto inputs = ncvslideio::gin(
            static_cast<ncvslideio::gapi::wip::IStreamSource::Ptr>(source));

    // The execution part
    pipeline.setSource(std::move(inputs));

    ncvslideio::TickMeter tm;
    ncvslideio::VideoWriter writer;

    ncvslideio::util::optional<ncvslideio::Mat> color_mask;
    ncvslideio::util::optional<ncvslideio::Mat> image;
    ncvslideio::Mat last_image;
    ncvslideio::Mat last_color_mask;

    pipeline.start();
    tm.start();

    std::size_t frames = 0u;
    std::size_t masks  = 0u;
    while (pipeline.pull(ncvslideio::gout(image, color_mask))) {
        if (image.has_value()) {
            ++frames;
            last_image = std::move(*image);
        }

        if (color_mask.has_value()) {
            ++masks;
            last_color_mask = std::move(*color_mask);
        }

        if (!last_image.empty() && !last_color_mask.empty()) {
            tm.stop();

            std::string stream_fps = "Stream FPS: " + toStr(frames / tm.getTimeSec());
            std::string inference_fps = "Inference FPS: " + toStr(masks  / tm.getTimeSec());

            ncvslideio::Mat tmp = last_image.clone();

            vis::drawResults(tmp, last_color_mask);
            vis::putText(tmp, {10, 22}, stream_fps);
            vis::putText(tmp, {10, 22 + 30}, inference_fps);

            ncvslideio::imshow("Out", tmp);
            ncvslideio::waitKey(1);
            if (!output.empty()) {
                if (!writer.isOpened()) {
                    const auto sz = ncvslideio::Size{tmp.cols, tmp.rows};
                    writer.open(output, ncvslideio::VideoWriter::fourcc('M','J','P','G'), 25.0, sz);
                    CV_Assert(writer.isOpened());
                }
                writer << tmp;
            }

            tm.start();
        }
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " ("
              << frames / tm.getTimeSec()<< " FPS)" << std::endl;
    return 0;
}
