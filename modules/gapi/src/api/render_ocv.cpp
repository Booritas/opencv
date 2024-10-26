#include <opencv2/imgproc.hpp>
#include <opencv2/gapi/render/render.hpp> // Kernel API's

#include "api/render_ocv.hpp"
#include "backends/render/ft_render.hpp"

namespace ncvslideio
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME Support `decim` mosaic parameter
inline void mosaic(ncvslideio::Mat& mat, const ncvslideio::Rect &rect, int cellSz)
{
    ncvslideio::Rect mat_rect(0, 0, mat.cols, mat.rows);
    auto intersection = mat_rect & rect;

    ncvslideio::Mat msc_roi = mat(intersection);

    bool has_crop_x = false;
    bool has_crop_y = false;

    int cols = msc_roi.cols;
    int rows = msc_roi.rows;

    if (msc_roi.cols % cellSz != 0)
    {
        has_crop_x = true;
        cols -= msc_roi.cols % cellSz;
    }

    if (msc_roi.rows % cellSz != 0)
    {
        has_crop_y = true;
        rows -= msc_roi.rows % cellSz;
    }

    ncvslideio::Mat cell_roi;
    for(int i = 0; i < rows; i += cellSz )
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(ncvslideio::Rect(j, i, cellSz, cellSz));
            cell_roi = ncvslideio::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(ncvslideio::Rect(cols, i, msc_roi.cols - cols, cellSz));
            cell_roi = ncvslideio::mean(cell_roi);
        }
    }

    if (has_crop_y)
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(ncvslideio::Rect(j, rows, cellSz, msc_roi.rows - rows));
            cell_roi = ncvslideio::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(ncvslideio::Rect(cols, rows, msc_roi.cols - cols, msc_roi.rows - rows));
            cell_roi = ncvslideio::mean(cell_roi);
        }
    }
}

inline void blendImage(const ncvslideio::Mat& img,
                       const ncvslideio::Mat& alpha,
                       const ncvslideio::Point& org,
                       ncvslideio::Mat background)
{
    GAPI_Assert(alpha.type() == CV_32FC1);
    GAPI_Assert(background.channels() == 3u);

    ncvslideio::Mat roi = background(ncvslideio::Rect(org, img.size()));
    ncvslideio::Mat img32f_w;
    ncvslideio::merge(std::vector<ncvslideio::Mat>(3, alpha), img32f_w);

    ncvslideio::Mat roi32f_w(roi.size(), CV_32FC3, ncvslideio::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    ncvslideio::Mat img32f, roi32f;
    if (img.type() == CV_32FC3) {
        img.copyTo(img32f);
    } else {
        img.convertTo(img32f, CV_32F, 1.0/255);
    }

    roi.convertTo(roi32f, CV_32F, 1.0/255);

    ncvslideio::multiply(img32f, img32f_w, img32f);
    ncvslideio::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
}

inline void blendTextMask(ncvslideio::Mat& img,
                          ncvslideio::Mat& mask,
                          const ncvslideio::Point& tl,
                          const ncvslideio::Scalar& color)
{
    mask.convertTo(mask, CV_32FC1, 1 / 255.0);
    ncvslideio::Mat color_mask;

    ncvslideio::merge(std::vector<ncvslideio::Mat>(3, mask), color_mask);
    ncvslideio::Scalar color32f = color / 255.0;
    ncvslideio::multiply(color_mask, color32f, color_mask);

    blendImage(color_mask, mask, tl, img);
}

inline void poly(ncvslideio::Mat& mat,
                 const ncvslideio::gapi::wip::draw::Poly& pp)
{
    std::vector<std::vector<ncvslideio::Point>> points{pp.points};
    ncvslideio::fillPoly(mat, points, pp.color, pp.lt, pp.shift);
}

struct BGR2YUVConverter
{
    ncvslideio::Scalar cvtColor(const ncvslideio::Scalar& bgr) const
    {
        double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
        double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
        double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;

        return {y, u, v};
    }

    void cvtImg(const ncvslideio::Mat& in, ncvslideio::Mat& out) { ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2YUV); }
};

struct EmptyConverter
{
    ncvslideio::Scalar cvtColor(const ncvslideio::Scalar& bgr)   const { return bgr; }
    void cvtImg(const ncvslideio::Mat& in, ncvslideio::Mat& out) const { out = in;   }
};

// FIXME util::visitor ?
template <typename ColorConverter>
void drawPrimitivesOCV(ncvslideio::Mat& in,
                       const ncvslideio::gapi::wip::draw::Prims& prims,
                       std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender>& ftpr)
{
    using namespace ncvslideio::gapi::wip::draw;

    ColorConverter converter;
    for (const auto &p : prims)
    {
        switch (p.index())
        {
            case Prim::index_of<Rect>():
            {
                const auto& rp = ncvslideio::util::get<Rect>(p);
                const auto color = converter.cvtColor(rp.color);
                ncvslideio::rectangle(in, rp.rect, color, rp.thick, rp.lt, rp.shift);
                break;
            }

            case Prim::index_of<Text>():
            {
                auto tp = ncvslideio::util::get<Text>(p);
                tp.color = converter.cvtColor(tp.color);
                ncvslideio::putText(in, tp.text, tp.org, tp.ff, tp.fs, tp.color, tp.thick, tp.lt, tp.bottom_left_origin);
                break;
            }

            case Prim::index_of<FText>():
            {
                const auto& ftp  = ncvslideio::util::get<FText>(p);
                const auto color = converter.cvtColor(ftp.color);

                GAPI_Assert(ftpr && "You must pass ncvslideio::gapi::wip::draw::freetype_font"
                                    " to the graph compile arguments");
                int baseline = 0;
                auto size    = ftpr->getTextSize(ftp.text, ftp.fh, &baseline);

                // Allocate mask outside
                ncvslideio::Mat mask(size, CV_8UC1, ncvslideio::Scalar::all(0));
                // Org it's bottom left position for baseline
                ncvslideio::Point org(0, mask.rows - baseline);
                ftpr->putText(mask, ftp.text, org, ftp.fh);

                // Org is bottom left point, transform it to top left point for blendImage
                ncvslideio::Point tl(ftp.org.x, ftp.org.y - mask.size().height + baseline);

                blendTextMask(in, mask, tl, color);
                break;
            }

            case Prim::index_of<Circle>():
            {
                const auto& cp = ncvslideio::util::get<Circle>(p);
                const auto color = converter.cvtColor(cp.color);
                ncvslideio::circle(in, cp.center, cp.radius, color, cp.thick, cp.lt, cp.shift);
                break;
            }

            case Prim::index_of<Line>():
            {
                const auto& lp = ncvslideio::util::get<Line>(p);
                const auto color = converter.cvtColor(lp.color);
                ncvslideio::line(in, lp.pt1, lp.pt2, color, lp.thick, lp.lt, lp.shift);
                break;
            }

            case Prim::index_of<Mosaic>():
            {
                const auto& mp = ncvslideio::util::get<Mosaic>(p);
                GAPI_Assert(mp.decim == 0 && "Only decim = 0 supported now");
                mosaic(in, mp.mos, mp.cellSz);
                break;
            }

            case Prim::index_of<Image>():
            {
                const auto& ip = ncvslideio::util::get<Image>(p);

                ncvslideio::Mat img;
                converter.cvtImg(ip.img, img);

                img.convertTo(img, CV_32FC1, 1.0 / 255);
                blendImage(img, ip.alpha, ip.org, in);
                break;
            }

            case Prim::index_of<Poly>():
            {
                auto pp = ncvslideio::util::get<Poly>(p);
                pp.color = converter.cvtColor(pp.color);
                poly(in, pp);
                break;
            }

            default: ncvslideio::util::throw_error(std::logic_error("Unsupported draw operation"));
        }
    }
}

void drawPrimitivesOCVBGR(ncvslideio::Mat                                                  &in,
                          const ncvslideio::gapi::wip::draw::Prims                         &prims,
                          std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender> &ftpr)
{
    drawPrimitivesOCV<EmptyConverter>(in, prims, ftpr);
}

void drawPrimitivesOCVYUV(ncvslideio::Mat                                                  &in,
                          const ncvslideio::gapi::wip::draw::Prims                         &prims,
                          std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender> &ftpr)
{
    drawPrimitivesOCV<BGR2YUVConverter>(in, prims, ftpr);
}

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace ncvslideio
