#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>
#include <tuple>
#include <memory>

#include <opencv2/imgproc.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gpu/ggpukernel.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/default.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/ocl/core.hpp>

const std::string about =
    "This is an example presents decoding on GPU using VPL Source and passing it to OpenCL backend";
const std::string keys =
    "{ h help       |                                                                | Print this help message }"
    "{ input        |                                                                | Path to the input video file. Use .avi extension }"
    "{ accel_mode   | mfxImplDescription.AccelerationMode:MFX_ACCEL_MODE_VIA_D3D11   | Acceleration mode for VPL }";

namespace {
namespace cfg {
// FIXME: Move OneVPL arguments parser to a single place
typename ncvslideio::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line);
} // namespace cfg
} // anonymous namespace

int main(int argc, char *argv[]) {
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    // Get file name
    const auto input = cmd.get<std::string>("input");
    const auto accel_mode = cmd.get<std::string>("accel_mode");

    // Create VPL config
    std::vector<ncvslideio::gapi::wip::onevpl::CfgParam> source_cfgs;
    source_cfgs.push_back(cfg::create_from_string(accel_mode));

    // Create VPL-based source
    std::shared_ptr<ncvslideio::gapi::wip::onevpl::IDeviceSelector> default_device_selector =
                                                ncvslideio::gapi::wip::onevpl::getDefaultDeviceSelector(source_cfgs);

    ncvslideio::gapi::wip::IStreamSource::Ptr source = ncvslideio::gapi::wip::make_onevpl_src(input, source_cfgs,
                                                                              default_device_selector);

    // Build the graph
    ncvslideio::GFrame in; // input frame from VPL source
    auto bgr_gmat = ncvslideio::gapi::streaming::BGR(in); // conversion from VPL source frame to BGR UMat
    auto out = ncvslideio::gapi::blur(bgr_gmat, ncvslideio::Size(4,4)); // ocl kernel of blur operation

    ncvslideio::GStreamingCompiled pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::core::ocl::kernels()));
    pipeline.setSource(std::move(source));

    // The execution part
    size_t frames = 0u;
    ncvslideio::TickMeter tm;
    ncvslideio::Mat outMat;

    pipeline.start();
    tm.start();

    while (pipeline.pull(ncvslideio::gout(outMat))) {
        ncvslideio::imshow("OutVideo", outMat);
        ncvslideio::waitKey(1);
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;

    return 0;
}

namespace {
namespace cfg {
typename ncvslideio::gapi::wip::onevpl::CfgParam create_from_string(const std::string &line) {
    using namespace ncvslideio::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return ncvslideio::gapi::wip::onevpl::CfgParam::create(name, value,
                                                   /* vpp params strongly optional */
                                                   name.find("vpp.") == std::string::npos);
}
} // namespace cfg
} // anonymous namespace
