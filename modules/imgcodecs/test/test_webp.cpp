// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_WEBP

TEST(Imgcodecs_WebP, encode_decode_lossless_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    string filename = root + "../ncvslideio/shared/lena.png";
    ncvslideio::Mat img = ncvslideio::imread(filename);
    ASSERT_FALSE(img.empty());

    string output = ncvslideio::tempfile(".webp");
    EXPECT_NO_THROW(ncvslideio::imwrite(output, img)); // lossless

    ncvslideio::Mat img_webp = ncvslideio::imread(output);

    std::vector<unsigned char> buf;

    FILE * wfile = NULL;

    wfile = fopen(output.c_str(), "rb");
    if (wfile != NULL)
    {
        fseek(wfile, 0, SEEK_END);
        size_t wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        buf.resize(wfile_size);

        size_t data_size = fread(&buf[0], 1, wfile_size, wfile);

        if(wfile)
        {
            fclose(wfile);
        }

        if (data_size != wfile_size)
        {
            EXPECT_TRUE(false);
        }
    }

    EXPECT_EQ(0, remove(output.c_str()));

    ncvslideio::Mat decode = ncvslideio::imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(decode.empty());
    EXPECT_TRUE(cvtest::norm(decode, img_webp, NORM_INF) == 0);

    ncvslideio::Mat decode_rgb = ncvslideio::imdecode(buf, IMREAD_COLOR_RGB);
    ASSERT_FALSE(decode_rgb.empty());

    cvtColor(decode_rgb, decode_rgb, COLOR_RGB2BGR);
    EXPECT_TRUE(cvtest::norm(decode_rgb, img_webp, NORM_INF) == 0);

    ASSERT_FALSE(img_webp.empty());

    EXPECT_TRUE(cvtest::norm(img, img_webp, NORM_INF) == 0);
}

TEST(Imgcodecs_WebP, encode_decode_lossy_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../ncvslideio/shared/lena.png";
    ncvslideio::Mat img = ncvslideio::imread(input);
    ASSERT_FALSE(img.empty());

    for(int q = 100; q>=0; q-=20)
    {
        std::vector<int> params;
        params.push_back(IMWRITE_WEBP_QUALITY);
        params.push_back(q);
        string output = ncvslideio::tempfile(".webp");

        EXPECT_NO_THROW(ncvslideio::imwrite(output, img, params));
        ncvslideio::Mat img_webp = ncvslideio::imread(output);
        EXPECT_EQ(0, remove(output.c_str()));
        EXPECT_FALSE(img_webp.empty());
        EXPECT_EQ(3,   img_webp.channels());
        EXPECT_EQ(512, img_webp.cols);
        EXPECT_EQ(512, img_webp.rows);
    }
}

TEST(Imgcodecs_WebP, encode_decode_with_alpha_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../ncvslideio/shared/lena.png";
    ncvslideio::Mat img = ncvslideio::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<ncvslideio::Mat> imgs;
    ncvslideio::split(img, imgs);
    imgs.push_back(ncvslideio::Mat(imgs[0]));
    imgs[imgs.size() - 1] = ncvslideio::Scalar::all(128);
    ncvslideio::merge(imgs, img);

    string output = ncvslideio::tempfile(".webp");

    EXPECT_NO_THROW(ncvslideio::imwrite(output, img));
    ncvslideio::Mat img_webp = ncvslideio::imread(output, IMREAD_UNCHANGED);
    ncvslideio::Mat img_webp_bgr = ncvslideio::imread(output); // IMREAD_COLOR by default
    EXPECT_EQ(0, remove(output.c_str()));
    EXPECT_FALSE(img_webp.empty());
    EXPECT_EQ(4,   img_webp.channels());
    EXPECT_EQ(512, img_webp.cols);
    EXPECT_EQ(512, img_webp.rows);
    EXPECT_FALSE(img_webp_bgr.empty());
    EXPECT_EQ(3,   img_webp_bgr.channels());
    EXPECT_EQ(512, img_webp_bgr.cols);
    EXPECT_EQ(512, img_webp_bgr.rows);
}

#endif // HAVE_WEBP

}} // namespace
