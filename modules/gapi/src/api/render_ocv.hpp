#include <vector>
#include "render_priv.hpp"
#include "backends/render/ft_render.hpp"

#ifndef OPENCV_RENDER_OCV_HPP
#define OPENCV_RENDER_OCV_HPP

namespace ncvslideio
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
void GAPI_EXPORTS drawPrimitivesOCVYUV(ncvslideio::Mat& yuv, const Prims& prims, std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender>& mc);
void GAPI_EXPORTS drawPrimitivesOCVBGR(ncvslideio::Mat& bgr, const Prims& prims, std::shared_ptr<ncvslideio::gapi::wip::draw::FTTextRender>& mc);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace ncvslideio

#endif // OPENCV_RENDER_OCV_HPP
