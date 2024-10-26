#include <fstream>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gframe.hpp>

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/streaming/format.hpp> // BGR accessor

#include <opencv2/highgui.hpp> // CommandLineParser

const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.h265  | Path to the output .h265 video file }";

int main(int argc, char *argv[]) {
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output_name = cmd.get<std::string>("output");

    ncvslideio::gapi::oak::EncoderConfig cfg;
    cfg.profile = ncvslideio::gapi::oak::EncoderConfig::Profile::H265_MAIN;

    ncvslideio::GFrame in;
    ncvslideio::GArray<uint8_t> encoded = ncvslideio::gapi::oak::encode(in, cfg);

    auto args = ncvslideio::compile_args(ncvslideio::gapi::oak::ColorCameraParams{}, ncvslideio::gapi::oak::kernels());

    auto pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(encoded)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::oak::ColorCamera>());
    pipeline.start();

    std::vector<uint8_t> out_h265_data;

    std::ofstream out_h265_file;
    out_h265_file.open(output_name, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 300;
    uint32_t pulled = 0;

    while (pipeline.pull(ncvslideio::gout(out_h265_data))) {
        if (out_h265_file.is_open()) {
            out_h265_file.write(reinterpret_cast<const char*>(out_h265_data.data()),
                                                              out_h265_data.size());
        }
        if (pulled++ == frames) {
            pipeline.stop();
            break;
        }
    }

    std::cout << "Pipeline finished: " << output_name << " file has been written." << std::endl;
}
