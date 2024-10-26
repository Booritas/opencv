#include <opencv2/imgproc.hpp> // ncvslideio::FONT*, ncvslideio::LINE*, ncvslideio::FILLED
#include <opencv2/highgui.hpp> // imwrite

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/render.hpp>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Filename required" << std::endl;
        return 1;
    }

    const auto font  = ncvslideio::FONT_HERSHEY_DUPLEX;
    const auto blue  = ncvslideio::Scalar{ 255,   0,   0}; // B/G/R
    const auto green = ncvslideio::Scalar{   0, 255,   0};
    const auto coral = ncvslideio::Scalar{0x81,0x81,0xF1};
    const auto white = ncvslideio::Scalar{ 255, 255, 255};
    ncvslideio::Mat test(ncvslideio::Size(480, 160), CV_8UC3, white);

    namespace draw = ncvslideio::gapi::wip::draw;
    std::vector<draw::Prim> prims;
    prims.emplace_back(draw::Circle{   // CIRCLE primitive
            {400,72},                  // Position (a ncvslideio::Point)
            32,                        // Radius
            coral,                     // Color
            ncvslideio::FILLED,                // Thickness/fill type
            ncvslideio::LINE_8,                // Line type
            0                          // Shift
        });
    prims.emplace_back(draw::Text{     // TEXT primitive
            "Hello from G-API!",       // Text
            {64,96},                   // Position (a ncvslideio::Point)
            font,                      // Font
            1.0,                       // Scale (size)
            blue,                      // Color
            2,                         // Thickness
            ncvslideio::LINE_8,                // Line type
            false                      // Bottom left origin flag
        });
    prims.emplace_back(draw::Rect{     // RECTANGLE primitive
            {16,48,400,72},            // Geometry (a ncvslideio::Rect)
            green,                     // Color
            2,                         // Thickness
            ncvslideio::LINE_8,                // Line type
            0                          // Shift
        });
    prims.emplace_back(draw::Mosaic{   // MOSAIC primitive
            {320,96,128,32},           // Geometry (a ncvslideio::Rect)
            16,                        // Cell size
            0                          // Decimation
        });
    draw::render(test, prims);
    ncvslideio::imwrite(argv[1], test);
    return 0;
}
