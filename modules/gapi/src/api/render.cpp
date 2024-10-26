#include "precomp.hpp"

#include <stdexcept>

#include <opencv2/gapi/render/render.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include "api/render_priv.hpp"

void ncvslideio::gapi::wip::draw::render(ncvslideio::Mat& bgr,
                                 const ncvslideio::gapi::wip::draw::Prims& prims,
                                 ncvslideio::GCompileArgs&& args)
{
    ncvslideio::GMat in;
    ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim> arr;

    ncvslideio::GComputation comp(ncvslideio::GIn(in, arr),
                          ncvslideio::GOut(ncvslideio::gapi::wip::draw::render3ch(in, arr)));
    comp.apply(ncvslideio::gin(bgr, prims), ncvslideio::gout(bgr), std::move(args));
}

void ncvslideio::gapi::wip::draw::render(ncvslideio::Mat& y_plane,
                                 ncvslideio::Mat& uv_plane,
                                 const Prims& prims,
                                 ncvslideio::GCompileArgs&& args)
{
    ncvslideio::GMat y_in, uv_in, y_out, uv_out;
    ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = ncvslideio::gapi::wip::draw::renderNV12(y_in, uv_in, arr);

    ncvslideio::GComputation comp(ncvslideio::GIn(y_in, uv_in, arr), ncvslideio::GOut(y_out, uv_out));
    comp.apply(ncvslideio::gin(y_plane, uv_plane, prims),
               ncvslideio::gout(y_plane, uv_plane), std::move(args));
}

void ncvslideio::gapi::wip::draw::render(ncvslideio::MediaFrame& frame,
                                 const Prims& prims,
                                 ncvslideio::GCompileArgs&& args)
{
    ncvslideio::GFrame in, out;
    ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim> arr;
    out = ncvslideio::gapi::wip::draw::renderFrame(in, arr);

    ncvslideio::GComputation comp(ncvslideio::GIn(in, arr), ncvslideio::GOut(out));
    comp.apply(ncvslideio::gin(frame, prims),
               ncvslideio::gout(frame), std::move(args));
}


void ncvslideio::gapi::wip::draw::cvtYUVToNV12(const ncvslideio::Mat& yuv,
                                       ncvslideio::Mat& y,
                                       ncvslideio::Mat& uv)
{
    GAPI_Assert(yuv.size().width  % 2 == 0);
    GAPI_Assert(yuv.size().height % 2 == 0);

    std::vector<ncvslideio::Mat> chs(3);
    ncvslideio::split(yuv, chs);
    y = chs[0];
    ncvslideio::merge(std::vector<ncvslideio::Mat>{chs[1], chs[2]}, uv);
    ncvslideio::resize(uv, uv, uv.size() / 2, ncvslideio::INTER_LINEAR);
}

void ncvslideio::gapi::wip::draw::cvtNV12ToYUV(const ncvslideio::Mat& y,
                                       const ncvslideio::Mat& uv,
                                       ncvslideio::Mat& yuv)
{
    ncvslideio::Mat upsample_uv;
    ncvslideio::resize(uv, upsample_uv, uv.size() * 2, ncvslideio::INTER_LINEAR);
    ncvslideio::merge(std::vector<ncvslideio::Mat>{y, upsample_uv}, yuv);
}

ncvslideio::GMat ncvslideio::gapi::wip::draw::render3ch(const ncvslideio::GMat& src,
                                        const ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>& prims)
{
    return ncvslideio::gapi::wip::draw::GRenderBGR::on(src, prims);
}

std::tuple<ncvslideio::GMat, ncvslideio::GMat>
ncvslideio::gapi::wip::draw::renderNV12(const ncvslideio::GMat& y,
                                const ncvslideio::GMat& uv,
                                const ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>& prims)
{
    return ncvslideio::gapi::wip::draw::GRenderNV12::on(y, uv, prims);
}

ncvslideio::GFrame ncvslideio::gapi::wip::draw::renderFrame(const ncvslideio::GFrame& frame,
                                                const ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>& prims)
{
    return ncvslideio::gapi::wip::draw::GRenderFrame::on(frame, prims);
}
