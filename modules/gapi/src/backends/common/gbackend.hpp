// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_GBACKEND_HPP
#define OPENCV_GAPI_GBACKEND_HPP

#include <string>
#include <memory>

#include <ade/node.hpp>

#include "opencv2/gapi/garg.hpp"

#include "opencv2/gapi/util/optional.hpp"

#include "compiler/gmodel.hpp"

namespace ncvslideio {
namespace gimpl {

    inline ncvslideio::Mat asMat(RMat::View& v) {
#if !defined(GAPI_STANDALONE)
        if (v.dims().empty()) {
            return ncvslideio::Mat(v.rows(), v.cols(), v.type(), v.ptr(), v.step());
        } else {
            ncvslideio::Mat m(v.dims(), v.type(), v.ptr(), v.steps().data());
            if (v.dims().size() == 1) {
                // FIXME: ncvslideio::Mat() constructor will set m.dims to 2;
                // To obtain 1D Mat, we have to set m.dims back to 1 manually
                m.dims = 1;
            }
            return m;
        }
#else
        // FIXME: add a check that steps are default
        return v.dims().empty() ? ncvslideio::Mat(v.rows(), v.cols(), v.type(), v.ptr(), v.step())
                                : ncvslideio::Mat(v.dims(), v.type(), v.ptr());

#endif
    }
    inline RMat::View asView(const Mat& m, RMat::View::DestroyCallback&& cb = nullptr) {
#if !defined(GAPI_STANDALONE)
        RMat::View::stepsT steps(m.dims);
        for (int i = 0; i < m.dims; i++) {
            steps[i] = m.step[i];
        }
        return RMat::View(ncvslideio::descr_of(m), m.data, steps, std::move(cb));
#else
        return m.dims.empty()
            ? RMat::View(ncvslideio::descr_of(m), m.data, m.step, std::move(cb))
            // Own Mat doesn't support n-dimensional steps so default ones are used in this case
            : RMat::View(ncvslideio::descr_of(m), m.data, RMat::View::stepsT{}, std::move(cb));
#endif
    }

    class RMatOnMat : public RMat::IAdapter {
        ncvslideio::Mat m_mat;
    public:
        const void* data() const { return m_mat.data; }
        RMatOnMat(ncvslideio::Mat m) : m_mat(m) {}
        virtual RMat::View access(RMat::Access) override { return asView(m_mat); }
        virtual ncvslideio::GMatDesc desc() const override { return ncvslideio::descr_of(m_mat); }
    };

    // Forward declarations
    struct Data;
    struct RcDesc;

    struct GAPI_EXPORTS RMatMediaFrameAdapter final: public ncvslideio::RMat::IAdapter
    {
        using MapDescF = std::function<ncvslideio::GMatDesc(const GFrameDesc&)>;
        using MapDataF = std::function<ncvslideio::Mat(const GFrameDesc&, const ncvslideio::MediaFrame::View&)>;

        RMatMediaFrameAdapter(const ncvslideio::MediaFrame& frame,
                              const MapDescF& frameDescToMatDesc,
                              const MapDataF& frameViewToMat) :
            m_frame(frame),
            m_frameDesc(frame.desc()),
            m_frameDescToMatDesc(frameDescToMatDesc),
            m_frameViewToMat(frameViewToMat)
        { }

        virtual ncvslideio::RMat::View access(ncvslideio::RMat::Access a) override
        {
            auto rmatToFrameAccess = [](ncvslideio::RMat::Access rmatAccess) {
                switch(rmatAccess) {
                    case ncvslideio::RMat::Access::R:
                        return ncvslideio::MediaFrame::Access::R;
                    case ncvslideio::RMat::Access::W:
                        return ncvslideio::MediaFrame::Access::W;
                    default:
                        ncvslideio::util::throw_error(std::logic_error("ncvslideio::RMat::Access::R or "
                            "ncvslideio::RMat::Access::W can only be mapped to ncvslideio::MediaFrame::Access!"));
                }
            };

            auto fv = m_frame.access(rmatToFrameAccess(a));

            auto fvHolder = std::make_shared<ncvslideio::MediaFrame::View>(std::move(fv));
            auto callback = [fvHolder]() mutable { fvHolder.reset(); };

            return asView(m_frameViewToMat(m_frame.desc(), *fvHolder), callback);
        }

        virtual ncvslideio::GMatDesc desc() const override
        {
            return m_frameDescToMatDesc(m_frameDesc);
        }

        ncvslideio::MediaFrame m_frame;
        ncvslideio::GFrameDesc m_frameDesc;
        MapDescF m_frameDescToMatDesc;
        MapDataF m_frameViewToMat;
    };


namespace magazine {
    template<typename... Ts> struct Class
    {
        template<typename T> using MapT = std::unordered_map<int, T>;
        using MapM = std::unordered_map<int, GRunArg::Meta>;

        template<typename T>       MapT<T>& slot()
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
        template<typename T> const MapT<T>& slot() const
        {
            return std::get<ade::util::type_list_index<T, Ts...>::value>(slots);
        }
        template<typename T> MapM& meta()
        {
            return metas[ade::util::type_list_index<T, Ts...>::value];
        }
        template<typename T> const MapM& meta() const
        {
            return metas[ade::util::type_list_index<T, Ts...>::value];
        }
    private:
        std::tuple<MapT<Ts>...> slots;
        std::array<MapM, sizeof...(Ts)> metas;
    };

} // namespace magazine

using Mag = magazine::Class< ncvslideio::Mat
                           , ncvslideio::Scalar
                           , ncvslideio::detail::VectorRef
                           , ncvslideio::detail::OpaqueRef
                           , ncvslideio::RMat
                           , ncvslideio::RMat::View
                           , ncvslideio::MediaFrame
#if !defined(GAPI_STANDALONE)
                           , ncvslideio::UMat
#endif
                           >;

namespace magazine
{
    enum class HandleRMat { BIND, SKIP };
    // Extracts a memory object from GRunArg, stores it in appropriate slot in a magazine
    // Note:
    // Only RMats are expected here as a memory object for GMat shape.
    // If handleRMat is BIND, RMat will be accessed, and RMat::View and wrapping ncvslideio::Mat
    // will be placed into the magazine.
    // If handleRMat is SKIP, this function skips'RMat handling assuming that backend will do it on its own.
    // FIXME?
    // handleRMat parameter might be redundant if all device specific backends implement own bind routines
    // without utilizing magazine at all
    void GAPI_EXPORTS bindInArg (Mag& mag, const RcDesc &rc, const GRunArg  &arg, HandleRMat handleRMat = HandleRMat::BIND);

    // Extracts a memory object reference from GRunArgP, stores it in appropriate slot in a magazine
    // Note on RMat handling from bindInArg above is also applied here
    void GAPI_EXPORTS bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, HandleRMat handleRMat = HandleRMat::BIND);

    void         resetInternalData(Mag& mag, const Data &d);
    ncvslideio::GRunArg  getArg    (const Mag& mag, const RcDesc &ref);
    ncvslideio::GRunArgP getObjPtr (      Mag& mag, const RcDesc &rc, bool is_umat = false);
    void         writeBack (const Mag& mag, const RcDesc &rc, GRunArgP &g_arg);

    // A mandatory clean-up procedure to force proper lifetime of wrappers (ncvslideio::Mat, ncvslideio::RMat::View)
    // over not-owned data
    // FIXME? Add an RAII wrapper for that?
    // Or put objects which need to be cleaned-up into a separate stack allocated magazine?
    void         unbind(Mag &mag, const RcDesc &rc);
} // namespace magazine

namespace detail
{
template<typename... Ts> struct magazine
{
    template<typename T> using MapT = std::unordered_map<int, T>;
    template<typename T>       MapT<T>& slot()
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
    template<typename T> const MapT<T>& slot() const
    {
        return std::get<util::type_list_index<T, Ts...>::value>(slots);
    }
private:
    std::tuple<MapT<Ts>...> slots;
};
} // namespace detail

struct GRuntimeArgs
{
    GRunArgs   inObjs;
    GRunArgsP outObjs;
};

template<typename T>
inline ncvslideio::util::optional<T> getCompileArg(const ncvslideio::GCompileArgs &args)
{
    return ncvslideio::gapi::getCompileArg<T>(args);
}

void GAPI_EXPORTS createMat(const ncvslideio::GMatDesc& desc, ncvslideio::Mat& mat);

inline void convertInt64ToInt32(const int64_t* src, int* dst, size_t size)
{
    std::transform(src, src + size, dst,
                   [](int64_t el) { return static_cast<int>(el); });
}

inline void convertInt32ToInt64(const int* src, int64_t* dst, size_t size)
{
    std::transform(src, src + size, dst,
                   [](int el) { return static_cast<int64_t>(el); });
}

}} // ncvslideio::gimpl

#endif // OPENCV_GAPI_GBACKEND_HPP
