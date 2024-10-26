#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

int main(int argc, char *argv[])
{
    ncvslideio::VideoCapture cap;
    if (argc > 1) cap.open(argv[1]);
    else cap.open(0);
    CV_Assert(cap.isOpened());

    ncvslideio::GMat in;
    ncvslideio::GMat vga      = ncvslideio::gapi::resize(in, ncvslideio::Size(), 0.5, 0.5);
    ncvslideio::GMat gray     = ncvslideio::gapi::BGR2Gray(vga);
    ncvslideio::GMat blurred  = ncvslideio::gapi::blur(gray, ncvslideio::Size(5,5));
    ncvslideio::GMat edges    = ncvslideio::gapi::Canny(blurred, 32, 128, 3);
    ncvslideio::GMat b,g,r;
    std::tie(b,g,r)   = ncvslideio::gapi::split3(vga);
    ncvslideio::GMat out      = ncvslideio::gapi::merge3(b, g | edges, r);
    ncvslideio::GComputation ac(in, out);

    ncvslideio::Mat input_frame;
    ncvslideio::Mat output_frame;
    CV_Assert(cap.read(input_frame));
    do
    {
        ac.apply(input_frame, output_frame);
        ncvslideio::imshow("output", output_frame);
    } while (cap.read(input_frame) && ncvslideio::waitKey(30) < 0);

    return 0;
}
