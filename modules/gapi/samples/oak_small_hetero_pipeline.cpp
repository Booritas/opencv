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

    std::vector<int> h = {1, 0, -1,
                          2, 0, -2,
                          1, 0, -1};
    std::vector<int> v = { 1,  2,  1,
                           0,  0,  0,
                          -1, -2, -1};
    ncvslideio::Mat hk(3, 3, CV_32SC1, h.data());
    ncvslideio::Mat vk(3, 3, CV_32SC1, v.data());

    // Heterogeneous pipeline:
    // OAK camera -> Sobel -> streaming accessor (CPU)
    ncvslideio::GFrame in;
    ncvslideio::GFrame sobel = ncvslideio::gapi::oak::sobelXY(in, hk, vk);
    // Default camera and then sobel work only with nv12 format
    ncvslideio::GMat out = ncvslideio::gapi::streaming::Y(sobel);

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
