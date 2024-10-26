#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/gapi/imgproc.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    bool need_first_conversion  = true;
    bool need_second_conversion = false;

    ncvslideio::Size szOut(4, 4);
    ncvslideio::GComputation cc([&](){
// ! [GIOProtoArgs usage]
        auto ins = ncvslideio::GIn();
        ncvslideio::GMat in1;
        if (need_first_conversion)
            ins += ncvslideio::GIn(in1);

        ncvslideio::GMat in2;
        if (need_second_conversion)
            ins += ncvslideio::GIn(in2);

        auto outs = ncvslideio::GOut();
        ncvslideio::GMat out1 = ncvslideio::gapi::resize(in1, szOut);
        if (need_first_conversion)
            outs += ncvslideio::GOut(out1);

        ncvslideio::GMat out2 = ncvslideio::gapi::resize(in2, szOut);
        if (need_second_conversion)
            outs += ncvslideio::GOut(out2);
// ! [GIOProtoArgs usage]
        return ncvslideio::GComputation(std::move(ins), std::move(outs));
    });

// ! [GRunArgs usage]
    auto in_vector = ncvslideio::gin();

    ncvslideio::Mat in_mat1( 8,  8, CV_8UC3);
    ncvslideio::Mat in_mat2(16, 16, CV_8UC3);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    if (need_first_conversion)
        in_vector += ncvslideio::gin(in_mat1);
    if (need_second_conversion)
        in_vector += ncvslideio::gin(in_mat2);
// ! [GRunArgs usage]

// ! [GRunArgsP usage]
    auto out_vector = ncvslideio::gout();
    ncvslideio::Mat out_mat1, out_mat2;
    if (need_first_conversion)
        out_vector += ncvslideio::gout(out_mat1);
    if (need_second_conversion)
        out_vector += ncvslideio::gout(out_mat2);
// ! [GRunArgsP usage]

    auto stream = cc.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::imgproc::cpu::kernels()));
    stream.setSource(std::move(in_vector));

    stream.start();
    stream.pull(std::move(out_vector));
    stream.stop();

    return 0;
}
