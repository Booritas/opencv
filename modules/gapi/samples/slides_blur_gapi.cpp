#include <opencv2/gapi.hpp>                            // G-API framework header
#include <opencv2/gapi/imgproc.hpp>                    // ncvslideio::gapi::blur()
#include <opencv2/highgui.hpp>                         // ncvslideio::imread/imwrite

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;

    ncvslideio::GMat in;                                       // Express the graph:
    ncvslideio::GMat out = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));  // `out` is a result of `blur` of `in`

    ncvslideio::Mat in_mat = ncvslideio::imread(argv[1]);              // Get the real data
    ncvslideio::Mat out_mat;                                   // Output buffer (may be empty)

    ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))       // Declare a graph from `in` to `out`
        .apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));    // ...and run it immediately

    ncvslideio::imwrite(argv[2], out_mat);                     // Save the result
    return 0;
}
