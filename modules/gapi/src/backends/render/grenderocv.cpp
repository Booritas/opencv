#include <opencv2/imgproc.hpp>

#include "api/render_ocv.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>

struct RenderOCVState
{
    std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender> ftpr;
};

GAPI_OCV_KERNEL_ST(RenderBGROCVImpl, ncvslideio::gapi::wip::draw::GRenderBGR, RenderOCVState)
{
    static void run(const ncvslideio::Mat& in,
                    const ncvslideio::gapi::wip::draw::Prims& prims,
                    ncvslideio::Mat& out,
                    RenderOCVState& state)
    {
        // NB: If in and out ncvslideio::Mats are the same object
        // we can avoid copy and render on out ncvslideio::Mat
        // It's work if this kernel is last operation in the graph
        if (in.data != out.data) {
            in.copyTo(out);
        }

        ncvslideio::gapi::wip::draw::drawPrimitivesOCVBGR(out, prims, state.ftpr);
    }

    static void setup(const ncvslideio::GMatDesc& /* in */,
                      const ncvslideio::GArrayDesc& /* prims */,
                      std::shared_ptr<RenderOCVState>& state,
                      const ncvslideio::GCompileArgs& args)
    {
        using namespace ncvslideio::gapi::wip::draw;
        auto opt_freetype_font = ncvslideio::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (opt_freetype_font.has_value())
        {
            state->ftpr = std::make_shared<FTTextRender>(opt_freetype_font->path);
        }
    }
};

GAPI_OCV_KERNEL_ST(RenderNV12OCVImpl, ncvslideio::gapi::wip::draw::GRenderNV12, RenderOCVState)
{
    static void run(const ncvslideio::Mat& in_y,
                    const ncvslideio::Mat& in_uv,
                    const ncvslideio::gapi::wip::draw::Prims& prims,
                    ncvslideio::Mat& out_y,
                    ncvslideio::Mat& out_uv,
                    RenderOCVState& state)
    {
        // NB: If in and out ncvslideio::Mats are the same object
        // we can avoid copy and render on out ncvslideio::Mat
        // It's work if this kernel is last operation in the graph
        if (in_y.data != out_y.data) {
            in_y.copyTo(out_y);
        }

        if (in_uv.data != out_uv.data) {
            in_uv.copyTo(out_uv);
        }

        /* FIXME How to render correctly on NV12 format ?
         *
         * Rendering on NV12 via OpenCV looks like this:
         *
         * y --------> 1)(NV12 -> YUV) -> yuv -> 2)draw -> yuv -> 3)split -------> out_y
         *                  ^                                         |
         *                  |                                         |
         * uv --------------                                          `----------> out_uv
         *
         *
         * 1) Collect yuv mat from two planes, uv plain in two times less than y plane
         *    so, upsample uv in tow times, with bilinear interpolation
         *
         * 2) Render primitives on YUV
         *
         * 3) Convert yuv to NV12 (using bilinear interpolation)
         *
         */

        // NV12 -> YUV
        ncvslideio::Mat upsample_uv, yuv;
        ncvslideio::resize(in_uv, upsample_uv, in_uv.size() * 2, ncvslideio::INTER_LINEAR);
        ncvslideio::merge(std::vector<ncvslideio::Mat>{in_y, upsample_uv}, yuv);

        ncvslideio::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);

        // YUV -> NV12
        ncvslideio::Mat out_u, out_v, uv_plane;
        std::vector<ncvslideio::Mat> chs = {out_y, out_u, out_v};
        ncvslideio::split(yuv, chs);
        ncvslideio::merge(std::vector<ncvslideio::Mat>{chs[1], chs[2]}, uv_plane);
        ncvslideio::resize(uv_plane, out_uv, uv_plane.size() / 2, ncvslideio::INTER_LINEAR);
    }

    static void setup(const ncvslideio::GMatDesc&   /* in_y  */,
                      const ncvslideio::GMatDesc&   /* in_uv */,
                      const ncvslideio::GArrayDesc& /* prims */,
                      std::shared_ptr<RenderOCVState>& state,
                      const ncvslideio::GCompileArgs& args)
    {
        using namespace ncvslideio::gapi::wip::draw;
        auto has_freetype_font = ncvslideio::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (has_freetype_font)
        {
            state->ftpr = std::make_shared<FTTextRender>(has_freetype_font->path);
        }
    }
};

GAPI_OCV_KERNEL_ST(RenderFrameOCVImpl, ncvslideio::gapi::wip::draw::GRenderFrame, RenderOCVState)
{
    static void run(const ncvslideio::MediaFrame & in,
                    const ncvslideio::gapi::wip::draw::Prims & prims,
                    ncvslideio::MediaFrame & out,
                    RenderOCVState & state)
    {
        GAPI_Assert(in.desc().fmt == ncvslideio::MediaFormat::NV12);

        // FIXME: consider a better approach (aka native inplace operation)
        // Non-intuitive logic with shared_ptr Priv class
        out = in;

        auto desc = out.desc();
        ncvslideio::Mat upsample_uv, yuv;
        {
            auto r_in = in.access(ncvslideio::MediaFrame::Access::R);

            auto in_y = ncvslideio::Mat(desc.size, CV_8UC1, r_in.ptr[0], r_in.stride[0]);
            auto in_uv = ncvslideio::Mat(desc.size / 2, CV_8UC2, r_in.ptr[1], r_in.stride[1]);

        /* FIXME How to render correctly on NV12 format ?
         *
         * Rendering on NV12 via OpenCV looks like this:
         *
         * y --------> 1)(NV12 -> YUV) -> yuv -> 2)draw -> yuv -> 3)split -------> out_y
         *                  ^                                         |
         *                  |                                         |
         * uv --------------                                          `----------> out_uv
         *
         *
         * 1) Collect yuv mat from two planes, uv plain in two times less than y plane
         *    so, upsample uv in two times, with bilinear interpolation
         *
         * 2) Render primitives on YUV
         *
         * 3) Convert yuv to NV12 (using bilinear interpolation)
         *
         */

            // NV12 -> YUV
            ncvslideio::resize(in_uv, upsample_uv, in_uv.size() * 2, ncvslideio::INTER_LINEAR);
            ncvslideio::merge(std::vector<ncvslideio::Mat>{in_y, upsample_uv}, yuv);
        }

        ncvslideio::gapi::wip::draw::drawPrimitivesOCVYUV(yuv, prims, state.ftpr);

        // YUV -> NV12
        {
            auto w_out = out.access(ncvslideio::MediaFrame::Access::W);

            auto out_y = ncvslideio::Mat(desc.size, CV_8UC1, w_out.ptr[0], w_out.stride[0]);
            auto out_uv = ncvslideio::Mat(desc.size / 2, CV_8UC2, w_out.ptr[1], w_out.stride[1]);

            ncvslideio::Mat out_u, out_v, uv_plane;
            std::vector<ncvslideio::Mat> chs = { out_y, out_u, out_v };
            ncvslideio::split(yuv, chs);
            ncvslideio::merge(std::vector<ncvslideio::Mat>{chs[1], chs[2]}, uv_plane);
            ncvslideio::resize(uv_plane, out_uv, uv_plane.size() / 2, ncvslideio::INTER_LINEAR);
        }
    }

    static void setup(const ncvslideio::GFrameDesc&   /* in_nv12  */,
        const ncvslideio::GArrayDesc& /* prims */,
        std::shared_ptr<RenderOCVState>&state,
        const ncvslideio::GCompileArgs & args)
    {
        using namespace ncvslideio::gapi::wip::draw;
        auto has_freetype_font = ncvslideio::gapi::getCompileArg<freetype_font>(args);
        state = std::make_shared<RenderOCVState>();

        if (has_freetype_font)
        {
            state->ftpr = std::make_shared<FTTextRender>(has_freetype_font->path);
        }
    }
};


ncvslideio::GKernelPackage ncvslideio::gapi::render::ocv::kernels()
{
    const static auto pkg = ncvslideio::gapi::kernels<RenderBGROCVImpl, RenderNV12OCVImpl, RenderFrameOCVImpl>();
    return pkg;
}
