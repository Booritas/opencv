// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <opencv2/gapi/s11n.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <ade/util/zip_range.hpp>

namespace opencv_test
{

  namespace
  {
      G_TYPED_KERNEL(CustomResize, <ncvslideio::GMat(ncvslideio::GMat, ncvslideio::Size, double, double, int)>, "org.opencv.customk.resize")
      {
          static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, ncvslideio::Size sz, double fx, double fy, int) {
              if (sz.width != 0 && sz.height != 0)
              {
                  return in.withSize(sz);
              }
              else
              {
                  GAPI_Assert(fx != 0. && fy != 0.);
                  return in.withSize
                    (ncvslideio::Size(static_cast<int>(std::round(in.size.width  * fx)),
                                         static_cast<int>(std::round(in.size.height * fy))));
              }
          }
      };

      GAPI_OCV_KERNEL(CustomResizeImpl, CustomResize)
      {
          static void run(const ncvslideio::Mat& in, ncvslideio::Size sz, double fx, double fy, int interp, ncvslideio::Mat &out)
          {
              ncvslideio::resize(in, out, sz, fx, fy, interp);
          }
      };

      struct GComputationApplyTest: public ::testing::Test
      {
          ncvslideio::GMat in;
          ncvslideio::Mat  in_mat;
          ncvslideio::Mat  out_mat;
          ncvslideio::GComputation m_c;

          GComputationApplyTest() : in_mat(300, 300, CV_8UC1),
                                    m_c(ncvslideio::GIn(in), ncvslideio::GOut(CustomResize::on(in, ncvslideio::Size(100, 100),
                                                                               0.0, 0.0, ncvslideio::INTER_LINEAR)))
          {
          }
      };

      struct GComputationVectorMatsAsOutput: public ::testing::Test
      {
          ncvslideio::Mat  in_mat;
          ncvslideio::GComputation m_c;
          std::vector<ncvslideio::Mat> ref_mats;

          GComputationVectorMatsAsOutput() : in_mat(300, 300, CV_8UC3),
          m_c([&](){
                      ncvslideio::GMat in;
                      ncvslideio::GMat out[3];
                      std::tie(out[0], out[1], out[2]) = ncvslideio::gapi::split3(in);
                      return ncvslideio::GComputation({in}, {out[0], out[1], out[2]});
                  })
          {
              ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
              ncvslideio::split(in_mat, ref_mats);
          }

          void run(std::vector<ncvslideio::Mat>& out_mats)
          {
              m_c.apply({in_mat}, out_mats);
          }

          void check(const std::vector<ncvslideio::Mat>& out_mats)
          {
              for (const auto it : ade::util::zip(ref_mats, out_mats))
              {
                  const auto& ref_mat = std::get<0>(it);
                  const auto& out_mat = std::get<1>(it);

                  EXPECT_EQ(0, cvtest::norm(ref_mat, out_mat, NORM_INF));
              }
          }
      };

      struct GComputationPythonApplyTest: public ::testing::Test
      {
          ncvslideio::Size sz;
          MatType type;
          ncvslideio::Mat in_mat1, in_mat2, out_mat_ocv;
          ncvslideio::GComputation m_c;

          GComputationPythonApplyTest() : sz(ncvslideio::Size(300,300)), type(CV_8UC1),
          in_mat1(sz, type), in_mat2(sz, type), out_mat_ocv(sz, type),
          m_c([&](){
                  ncvslideio::GMat in1, in2;
                  ncvslideio::GMat out = in1 + in2;
                  return ncvslideio::GComputation(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
                  })
          {
              ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
              ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
              out_mat_ocv = in_mat1 + in_mat2;
          }
      };
  }

  TEST_F(GComputationPythonApplyTest, WithoutSerialization)
  {
      auto output = m_c.apply(ncvslideio::detail::ExtractArgsCallback{[this](const ncvslideio::GTypesInfo& info)
                                  {
                                      GAPI_Assert(info[0].shape == ncvslideio::GShape::GMAT);
                                      GAPI_Assert(info[1].shape == ncvslideio::GShape::GMAT);
                                      return ncvslideio::GRunArgs{in_mat1, in_mat2};
                                  }
                              });

      EXPECT_EQ(1u, output.size());

      const auto& out_mat_gapi = ncvslideio::util::get<ncvslideio::Mat>(output[0]);
      EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
  }

  TEST_F(GComputationPythonApplyTest, WithSerialization)
  {
      auto p = ncvslideio::gapi::serialize(m_c);
      auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

      auto output = c.apply(ncvslideio::detail::ExtractArgsCallback{[this](const ncvslideio::GTypesInfo& info)
                                  {
                                      GAPI_Assert(info[0].shape == ncvslideio::GShape::GMAT);
                                      GAPI_Assert(info[1].shape == ncvslideio::GShape::GMAT);
                                      return ncvslideio::GRunArgs{in_mat1, in_mat2};
                                  }
                              });

      EXPECT_EQ(1u, output.size());

      const auto& out_mat_gapi = ncvslideio::util::get<ncvslideio::Mat>(output[0]);
      EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
  }

  TEST_F(GComputationApplyTest, ThrowDontPassCustomKernel)
  {
      EXPECT_THROW(m_c.apply(in_mat, out_mat), std::logic_error);
  }

  TEST_F(GComputationApplyTest, NoThrowPassCustomKernel)
  {
      const auto pkg = ncvslideio::gapi::kernels<CustomResizeImpl>();

      ASSERT_NO_THROW(m_c.apply(in_mat, out_mat, ncvslideio::compile_args(pkg)));
  }

  TEST_F(GComputationVectorMatsAsOutput, OutputAllocated)
  {
      std::vector<ncvslideio::Mat> out_mats(3);
      for (auto& out_mat : out_mats)
      {
          out_mat.create(in_mat.size(), CV_8UC1);
      }

      run(out_mats);
      check(out_mats);
  }

  TEST_F(GComputationVectorMatsAsOutput, OutputNotAllocated)
  {
      std::vector<ncvslideio::Mat> out_mats(3);

      run(out_mats);
      check(out_mats);
  }

  TEST_F(GComputationVectorMatsAsOutput, OutputAllocatedWithInvalidMeta)
  {
      std::vector<ncvslideio::Mat> out_mats(3);

      for (auto& out_mat : out_mats)
      {
          out_mat.create(in_mat.size() / 2, CV_8UC1);
      }

      run(out_mats);
      check(out_mats);
  }

} // namespace opencv_test
