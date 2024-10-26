// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <ade/util/iota_range.hpp>
#include <ade/util/algorithm.hpp>

#include <opencv2/gapi/own/mat.hpp> //gapi::own::Mat
#include <opencv2/gapi/gmat.hpp>

#include "api/gorigin.hpp"

// ncvslideio::GMat public implementation //////////////////////////////////////////////
ncvslideio::GMat::GMat()
    : m_priv(new GOrigin(GShape::GMAT, GNode::Param()))
{
}

ncvslideio::GMat::GMat(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GMAT, n, out))
{
}

ncvslideio::GMat::GMat(ncvslideio::Mat m)
    : m_priv(new GOrigin(GShape::GMAT, ncvslideio::gimpl::ConstVal(m))) {
}

ncvslideio::GOrigin& ncvslideio::GMat::priv()
{
    return *m_priv;
}

const ncvslideio::GOrigin& ncvslideio::GMat::priv() const
{
    return *m_priv;
}

static std::vector<int> checkVectorImpl(const int width, const int height, const int chan,
                                        const int n)
{
    if (width == 1 && (n == -1 || n == chan))
    {
        return {height, chan};
    }
    else if (height == 1 && (n == -1 || n == chan))
    {
        return {width, chan};
    }
    else if (chan == 1 && (n == -1 || n == width))
    {
        return {height, width};
    }
    else // input Mat can't be described as vector of points of given dimensionality
    {
        return {-1, -1};
    }
}

int ncvslideio::gapi::detail::checkVector(const ncvslideio::GMatDesc& in, const size_t n)
{
    GAPI_Assert(n != 0u);
    return checkVectorImpl(in.size.width, in.size.height, in.chan, static_cast<int>(n))[0];
}

std::vector<int> ncvslideio::gapi::detail::checkVector(const ncvslideio::GMatDesc& in)
{
    return checkVectorImpl(in.size.width, in.size.height, in.chan, -1);
}

namespace{
    template <typename T> ncvslideio::GMetaArgs vec_descr_of(const std::vector<T> &vec)
        {
        ncvslideio::GMetaArgs vec_descr;
        vec_descr.reserve(vec.size());
        for(auto& mat : vec){
            vec_descr.emplace_back(descr_of(mat));
        }
        return vec_descr;
    }
}

#if !defined(GAPI_STANDALONE)
ncvslideio::GMatDesc ncvslideio::descr_of(const ncvslideio::Mat &mat)
{
    const auto mat_dims = mat.size.dims();

    if (mat_dims == 2)
        return GMatDesc{mat.depth(), mat.channels(), {mat.cols, mat.rows}};

    std::vector<int> dims(mat_dims);
    for (auto i : ade::util::iota(mat_dims)) {
        // Note: ncvslideio::MatSize is not iterable
        dims[i] = mat.size[i];
    }
    return GMatDesc{mat.depth(), std::move(dims)};
}
#endif

ncvslideio::GMatDesc ncvslideio::gapi::own::descr_of(const Mat &mat)
{
    return (mat.dims.empty())
        ? GMatDesc{mat.depth(), mat.channels(), {mat.cols, mat.rows}}
        : GMatDesc{mat.depth(), mat.dims};
}

#if !defined(GAPI_STANDALONE)
ncvslideio::GMatDesc ncvslideio::descr_of(const ncvslideio::UMat &mat)
{
    GAPI_Assert(mat.size.dims() == 2);
    return GMatDesc{ mat.depth(), mat.channels(),{ mat.cols, mat.rows } };
}

ncvslideio::GMetaArgs ncvslideio::descrs_of(const std::vector<ncvslideio::UMat> &vec)
{
    return vec_descr_of(vec);
}
#endif

ncvslideio::GMetaArgs ncvslideio::descrs_of(const std::vector<ncvslideio::Mat> &vec)
{
    return vec_descr_of(vec);
}

ncvslideio::GMetaArgs ncvslideio::gapi::own::descrs_of(const std::vector<Mat> &vec)
{
    return vec_descr_of(vec);
}

ncvslideio::GMatDesc ncvslideio::descr_of(const ncvslideio::RMat &mat)
{
    return mat.desc();
}

namespace ncvslideio {
std::ostream& operator<<(std::ostream& os, const ncvslideio::GMatDesc &desc)
{
    switch (desc.depth)
    {
#define TT(X) case CV_##X: os << #X; break;
        TT(8U);
        TT(8S);
        TT(16U);
        TT(16S);
        TT(32S);
        TT(32F);
        TT(64F);
#undef TT
    default:
        os << "(user type "
           << std::hex << desc.depth << std::dec
           << ")";
        break;
    }

    if (desc.isND()) {
        os << " [";
        for (size_t i = 0; i < desc.dims.size() - 1; ++i) {
            os << desc.dims[i] << "x";
        }
        os << desc.dims.back() << "]";
    } else {
        os << "C" << desc.chan;
        if (desc.planar) os << "p";
        os << " ";
        os << desc.size.width << "x" << desc.size.height;
    }

    return os;
}

namespace {
template<typename M> inline bool canDescribeHelper(const GMatDesc& desc, const M& mat)
{
    const auto mat_desc = desc.planar ? ncvslideio::descr_of(mat).asPlanar(desc.chan) : ncvslideio::descr_of(mat);
    return desc == mat_desc;
}
} // anonymous namespace

bool GMatDesc::canDescribe(const ncvslideio::Mat& mat) const
{
    return canDescribeHelper(*this, mat);
}

bool GMatDesc::canDescribe(const ncvslideio::RMat& mat) const
{
    return canDescribeHelper(*this, mat);
}

}// namespace ncvslideio
