// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifdef HAVE_FREETYPE
#include <codecvt>
#endif // HAVE_FREETYPE

#include "../test_precomp.hpp"
#include "../common/gapi_render_tests.hpp"

#include "api/render_priv.hpp"

namespace opencv_test
{

#ifdef HAVE_FREETYPE
GAPI_RENDER_TEST_FIXTURES(OCVTestFTexts,    FIXTURE_API(std::wstring, ncvslideio::Point, int, ncvslideio::Scalar),                        4, text, org, fh, color)
#endif // HAVE_FREETYPE

GAPI_RENDER_TEST_FIXTURES(OCVTestTexts,     FIXTURE_API(std::string, ncvslideio::Point, int, double, ncvslideio::Scalar, int, int, bool), 8, text, org, ff, fs, color, thick, lt, blo)
GAPI_RENDER_TEST_FIXTURES(OCVTestRects,     FIXTURE_API(ncvslideio::Rect, ncvslideio::Scalar, int, int, int),                             5, rect, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestCircles,   FIXTURE_API(ncvslideio::Point, int, ncvslideio::Scalar, int, int, int),                       6, center, radius, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestLines,     FIXTURE_API(ncvslideio::Point, ncvslideio::Point, ncvslideio::Scalar, int, int, int),                 6, pt1, pt2, color, thick, lt, shift)
GAPI_RENDER_TEST_FIXTURES(OCVTestMosaics,   FIXTURE_API(ncvslideio::Rect, int, int),                                              3, mos, cellsz, decim)
GAPI_RENDER_TEST_FIXTURES(OCVTestImages,    FIXTURE_API(ncvslideio::Rect, ncvslideio::Scalar, double),                                    3, rect, color, transparency)
GAPI_RENDER_TEST_FIXTURES(OCVTestPolylines, FIXTURE_API(Points, ncvslideio::Scalar, int, int, int),                               5, points, color, thick, lt, shift)

TEST_P(RenderBGROCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        ncvslideio::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Text{text, org, ff, fs, color, thick, lt, blo});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

class TestMediaNV12 final : public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_y;
    ncvslideio::Mat m_uv;
public:
    TestMediaNV12(ncvslideio::Mat y, ncvslideio::Mat uv) : m_y(y), m_uv(uv) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ ncvslideio::MediaFormat::NV12, ncvslideio::Size(m_y.cols, m_y.rows) };
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        ncvslideio::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
    }
};

TEST_P(RenderMFrameOCVTestTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Text{ text, org, ff, fs, color, thick, lt, blo });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}


# ifdef HAVE_FREETYPE

TEST_P(RenderBGROCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::FText{text, org, fh, color});
    EXPECT_NO_THROW(ncvslideio::gapi::wip::draw::render(gapi_mat, prims,
                                ncvslideio::compile_args(ncvslideio::gapi::wip::draw::freetype_font{
                                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                                })));
}

TEST_P(RenderNV12OCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::FText{text, org, fh, color});
    EXPECT_NO_THROW(ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims,
                                ncvslideio::compile_args(ncvslideio::gapi::wip::draw::freetype_font{
                                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
                                })));
}

TEST_P(RenderMFrameOCVTestFTexts, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::Mat y_copy_mat = y_gapi_mat.clone();
    ncvslideio::Mat uv_copy_mat = uv_gapi_mat.clone();
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::FText{ text, org, fh, color });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    EXPECT_NO_THROW(ncvslideio::gapi::wip::draw::render(nv12, prims,
        ncvslideio::compile_args(ncvslideio::gapi::wip::draw::freetype_font{
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            })));
    EXPECT_NE(0, ncvslideio::norm(y_gapi_mat, y_copy_mat));
    EXPECT_NE(0, ncvslideio::norm(uv_gapi_mat, uv_copy_mat));
}


static std::wstring to_wstring(const char* bytes)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.from_bytes(bytes);
}

TEST(RenderFText, FontsNotPassedToCompileArgs)
{
    ncvslideio::Mat in_mat(640, 480, CV_8UC3, ncvslideio::Scalar::all(0));

    std::wstring text = to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd");
    ncvslideio::Point org(100, 100);
    int fh = 60;
    ncvslideio::Scalar color(200, 100, 25);

    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::FText{text, org, fh, color});

    EXPECT_ANY_THROW(ncvslideio::gapi::wip::draw::render(in_mat, prims));
}

#endif // HAVE_FREETYPE

TEST_P(RenderBGROCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Rect{rect, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        ncvslideio::rectangle(ref_mat, rect, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Rect{rect, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestRects, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Rect{ rect, color, thick, lt, shift });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        ncvslideio::circle(ref_mat, center, radius, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Circle{center, radius, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::circle(yuv, center, radius, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestCircles, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Circle{ center, radius, color, thick, lt, shift });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::circle(yuv, center, radius, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        ncvslideio::line(ref_mat, pt1, pt2, color, thick, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Line{pt1, pt2, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::line(yuv, pt1, pt2, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestLines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Line{ pt1, pt2, color, thick, lt, shift });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::line(yuv, pt1, pt2, cvtBGRToYUVC(color), thick, lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Mosaic{mos, cellsz, decim});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        drawMosaicRef(ref_mat, mos, cellsz);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Mosaic{mos, cellsz, decim});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        drawMosaicRef(yuv, mos, cellsz);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestMosaics, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Mosaic{ mos, cellsz, decim });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        drawMosaicRef(yuv, mos, cellsz);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestImages, AccuracyTest)
{
    ncvslideio::Mat img(rect.size(), CV_8UC3, color);
    ncvslideio::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    ncvslideio::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Image{org, img, alpha});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        blendImageRef(ref_mat, org, img, alpha);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestImages, AccuracyTest)
{
    ncvslideio::Mat img(rect.size(), CV_8UC3, color);
    ncvslideio::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    ncvslideio::Point org = {tl.x, tl.y + rect.size().height};

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Image{org, img, alpha});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::Mat yuv_img;
        ncvslideio::cvtColor(img, yuv_img, ncvslideio::COLOR_BGR2YUV);
        blendImageRef(yuv, org, yuv_img, alpha);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestImages, AccuracyTest)
{
    ncvslideio::Mat img(rect.size(), CV_8UC3, color);
    ncvslideio::Mat alpha(rect.size(), CV_32FC1, transparency);
    auto tl = rect.tl();
    ncvslideio::Point org = { tl.x, tl.y + rect.size().height };

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Image{ org, img, alpha });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        ncvslideio::Mat yuv_img;
        ncvslideio::cvtColor(img, yuv_img, ncvslideio::COLOR_BGR2YUV);
        blendImageRef(yuv, org, yuv_img, alpha);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderBGROCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Poly{points, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        std::vector<std::vector<ncvslideio::Point>> array_points{points};
        ncvslideio::fillPoly(ref_mat, array_points, color, lt, shift);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(gapi_mat, ref_mat));
    }
}

TEST_P(RenderNV12OCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Poly{points, color, thick, lt, shift});
    ncvslideio::gapi::wip::draw::render(y_gapi_mat, uv_gapi_mat, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        std::vector<std::vector<ncvslideio::Point>> pp{points};
        ncvslideio::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat,  y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

TEST_P(RenderMFrameOCVTestPolylines, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Poly{ points, color, thick, lt, shift });
    ncvslideio::MediaFrame nv12 = ncvslideio::MediaFrame::Create<TestMediaNV12>(y_gapi_mat, uv_gapi_mat);
    ncvslideio::gapi::wip::draw::render(nv12, prims);

    // OpenCV code //////////////////////////////////////////////////////////////
    {
        // NV12 -> YUV
        ncvslideio::Mat yuv;
        ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

        std::vector<std::vector<ncvslideio::Point>> pp{ points };
        ncvslideio::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

        // YUV -> NV12
        ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, ncvslideio::norm(y_gapi_mat, y_ref_mat));
        EXPECT_EQ(0, ncvslideio::norm(uv_gapi_mat, uv_ref_mat));
    }
}

// FIXME avoid code duplicate for NV12 and BGR cases
INSTANTIATE_TEST_CASE_P(RenderBGROCVTestRectsImpl, RenderBGROCVTestRects,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200)),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8, LINE_4),
                                Values(0, 1)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestRectsImpl, RenderNV12OCVTestRects,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                       Values(ncvslideio::Rect(100, 100, 200, 200)),
                                       Values(ncvslideio::Scalar(100, 50, 150)),
                                       Values(2),
                                       Values(LINE_8),
                                       Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestRectsImpl, RenderMFrameOCVTestRects,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200)),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestCirclesImpl, RenderBGROCVTestCircles,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(10),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestCirclesImpl, RenderNV12OCVTestCircles,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(10),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8, LINE_4),
                                Values(0, 1)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestCirclesImpl, RenderMFrameOCVTestCircles,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(10),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestLinesImpl, RenderBGROCVTestLines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(ncvslideio::Point(200, 200)),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestLinesImpl, RenderNV12OCVTestLines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(ncvslideio::Point(200, 200)),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestLinesImpl, RenderMFrameOCVTestLines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Point(100, 100)),
                                Values(ncvslideio::Point(200, 200)),
                                Values(ncvslideio::Scalar(100, 50, 150)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestTextsImpl, RenderBGROCVTestTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values("SomeText"),
                                Values(ncvslideio::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(ncvslideio::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestTextsImpl, RenderNV12OCVTestTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values("SomeText"),
                                Values(ncvslideio::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(ncvslideio::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestTextsImpl, RenderMFrameOCVTestTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values("SomeText"),
                                Values(ncvslideio::Point(200, 200)),
                                Values(FONT_HERSHEY_SIMPLEX),
                                Values(2.0),
                                Values(ncvslideio::Scalar(0, 255, 0)),
                                Values(2),
                                Values(LINE_8),
                                Values(false)));


#ifdef HAVE_FREETYPE

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestFTextsImpl, RenderBGROCVTestFTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                            Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                   to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                            Values(ncvslideio::Point(200, 200)),
                            Values(64),
                            Values(ncvslideio::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestFTextsImpl, RenderNV12OCVTestFTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                            Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                   to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                            Values(ncvslideio::Point(200, 200)),
                            Values(64),
                            Values(ncvslideio::Scalar(0, 255, 0))));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestFTextsImpl, RenderMFrameOCVTestFTexts,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(to_wstring("\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c"),
                                to_wstring("\xe3\x80\xa4\xe3\x80\xa5\xe3\x80\xa6\xe3\x80\xa7\xe3\x80\xa8\xe3\x80\x85\xe3\x80\x86")),
                                Values(ncvslideio::Point(200, 200)),
                                Values(64),
                                Values(ncvslideio::Scalar(0, 255, 0))));

#endif // HAVE_FREETYPE

// FIXME Implement a macros to instantiate the tests because BGR and NV12 have the same parameters

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestMosaicsImpl, RenderBGROCVTestMosaics,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200),      // Normal case
                                       ncvslideio::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       ncvslideio::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       ncvslideio::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       ncvslideio::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       ncvslideio::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       ncvslideio::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       ncvslideio::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       ncvslideio::Rect(500, -50, 200, 200),      // Intersection with top side
                                       ncvslideio::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       ncvslideio::Rect(5000, 2000, 100, 100),    // Outside image
                                       ncvslideio::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       ncvslideio::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestMosaicsImpl, RenderNV12OCVTestMosaics,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200),      // Normal case
                                       ncvslideio::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       ncvslideio::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       ncvslideio::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       ncvslideio::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       ncvslideio::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       ncvslideio::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       ncvslideio::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       ncvslideio::Rect(500, -50, 200, 200),      // Intersection with top side
                                       ncvslideio::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       ncvslideio::Rect(5000, 2000, 100, 100),    // Outside image
                                       ncvslideio::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       ncvslideio::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestMosaicsImpl, RenderMFrameOCVTestMosaics,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200),      // Normal case
                                       ncvslideio::Rect(-50, -50, 200, 200),      // Intersection with left-top corner
                                       ncvslideio::Rect(-50, 100, 200, 200),      // Intersection with left side
                                       ncvslideio::Rect(-50, 600, 200, 200),      // Intersection with left-bottom corner
                                       ncvslideio::Rect(100, 600, 200, 200),      // Intersection with bottom side
                                       ncvslideio::Rect(1200, 700, 200, 200),     // Intersection with right-bottom corner
                                       ncvslideio::Rect(1200, 400, 200, 200),     // Intersection with right side
                                       ncvslideio::Rect(1200, -50, 200, 200),     // Intersection with right-top corner
                                       ncvslideio::Rect(500, -50, 200, 200),      // Intersection with top side
                                       ncvslideio::Rect(-100, 300, 1480, 300),    // From left to right side with intersection
                                       ncvslideio::Rect(5000, 2000, 100, 100),    // Outside image
                                       ncvslideio::Rect(-300, -300, 3000, 3000),  // Cover all image
                                       ncvslideio::Rect(100, 100, -500, -500)),   // Negative width and height
                                Values(25),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestImagesImpl, RenderBGROCVTestImages,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200)),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestImagesImpl, RenderNV12OCVTestImages,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200)),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestImagesImpl, RenderMFrameOCVTestImages,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(ncvslideio::Rect(100, 100, 200, 200)),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(1.0)));

INSTANTIATE_TEST_CASE_P(RenderBGROCVTestPolylinesImpl, RenderBGROCVTestPolylines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(std::vector<ncvslideio::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderNV12OCVTestPolylinesImpl, RenderNV12OCVTestPolylines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(std::vector<ncvslideio::Point>{{100, 100}, {200, 200}, {150, 300}, {400, 150}}),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));

INSTANTIATE_TEST_CASE_P(RenderMFrameOCVTestPolylinesImpl, RenderMFrameOCVTestPolylines,
                        Combine(Values(ncvslideio::Size(1280, 720)),
                                Values(std::vector<ncvslideio::Point>{ {100, 100}, { 200, 200 }, { 150, 300 }, { 400, 150 }}),
                                Values(ncvslideio::Scalar(100, 150, 60)),
                                Values(2),
                                Values(LINE_8),
                                Values(0)));
}
