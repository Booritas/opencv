// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP
#define GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP

#include <memory>
#include <vector>

#include "streaming/onevpl/engine/preproc_engine_interface.hpp"

namespace ncvslideio {
namespace gapi {
namespace wip {
namespace onevpl {

// GAPI_EXPORTS for tests
class GAPI_EXPORTS VPPPreprocDispatcher final : public ncvslideio::gapi::wip::IPreprocEngine {
public:

    ncvslideio::util::optional<pp_params> is_applicable(const ncvslideio::MediaFrame& in_frame) override;

    pp_session initialize_preproc(const pp_params& initial_frame_param,
                                  const GFrameDesc& required_frame_descr) override;

    ncvslideio::MediaFrame run_sync(const pp_session &session_handle,
                            const ncvslideio::MediaFrame& in_frame,
                            const ncvslideio::util::optional<ncvslideio::Rect> &opt_roi) override;

    template<class PreprocImpl, class ...Args>
    void insert_worker(Args&& ...args) {
        workers.emplace_back(std::make_shared<PreprocImpl>(std::forward<Args>(args)...));
    }

    size_t size() const {
        return workers.size();
    }
private:
    std::vector<std::shared_ptr<ncvslideio::gapi::wip::IPreprocEngine>> workers;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace ncvslideio
#endif // GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP
