/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_RGB_HISTOGRAM_HPP
#define VAS_OT_RGB_HISTOGRAM_HPP

#include <opencv2/core.hpp>
#include <cstdint>

namespace vas {
namespace ot {

class RgbHistogram {
  public:
    explicit RgbHistogram(int32_t rgb_bin_size);
    virtual ~RgbHistogram(void);

    virtual void Compute(const ncvslideio::Mat &image, ncvslideio::Mat *hist);
    virtual void ComputeFromBgra32(const ncvslideio::Mat &image, ncvslideio::Mat *hist);
    virtual int32_t FeatureSize(void) const; // currently 512 * float32

    static float ComputeSimilarity(const ncvslideio::Mat &hist1, const ncvslideio::Mat &hist2);

  protected:
    int32_t rgb_bin_size_;
    int32_t rgb_num_bins_;
    int32_t rgb_hist_size_;

    void AccumulateRgbHistogram(const ncvslideio::Mat &patch, float *rgb_hist) const;
    void AccumulateRgbHistogram(const ncvslideio::Mat &patch, const ncvslideio::Mat &weight, float *rgb_hist) const;

    void AccumulateRgbHistogramFromBgra32(const ncvslideio::Mat &patch, float *rgb_hist) const;
    void AccumulateRgbHistogramFromBgra32(const ncvslideio::Mat &patch, const ncvslideio::Mat &weight, float *rgb_hist) const;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_RGB_HISTOGRAM_HPP
