#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/media.hpp>

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/streaming/format.hpp> // BGR accessor

#include <opencv2/highgui.hpp> // CommandLineParser

const std::string keys =
    "{ h help  |              | Print this help message }"
    "{ output  | output.png   | Path to the output file }";

int main(int argc, char *argv[]) {
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    const std::string output_name = cmd.get<std::string>("output");

    ncvslideio::GFrame in;
    // Actually transfers data to host
    ncvslideio::GFrame copy = ncvslideio::gapi::oak::copy(in);
    // Default camera works only with nv12 format
    ncvslideio::GMat out = ncvslideio::gapi::streaming::Y(copy);

    auto args = ncvslideio::compile_args(ncvslideio::gapi::oak::ColorCameraParams{},
                                 ncvslideio::gapi::oak::kernels());

    auto pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    ncvslideio::Mat out_mat(1920, 1080, CV_8UC1);

    pipeline.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::oak::ColorCamera>());
    pipeline.start();

    // pull 1 frame
    pipeline.pull(ncvslideio::gout(out_mat));

    ncvslideio::imwrite(output_name, out_mat);

    std::cout << "Pipeline finished: " << output_name << " file has been written." << std::endl;
}
