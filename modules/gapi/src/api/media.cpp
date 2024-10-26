// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include <opencv2/gapi/media.hpp>

struct ncvslideio::MediaFrame::Priv {
    std::unique_ptr<IAdapter> adapter;
};

ncvslideio::MediaFrame::MediaFrame() {
}

ncvslideio::MediaFrame::MediaFrame(AdapterPtr &&ptr)
    : m(new Priv{std::move(ptr)}) {
}

ncvslideio::GFrameDesc ncvslideio::MediaFrame::desc() const {
    return m->adapter->meta();
}

ncvslideio::MediaFrame::View ncvslideio::MediaFrame::access(Access code) const {
    return m->adapter->access(code);
}

ncvslideio::util::any ncvslideio::MediaFrame::blobParams() const
{
    return m->adapter->blobParams();
}

ncvslideio::MediaFrame::IAdapter* ncvslideio::MediaFrame::getAdapter() const {
    return m->adapter.get();
}

void ncvslideio::MediaFrame::serialize(ncvslideio::gapi::s11n::IOStream& os) const {
    m->adapter->serialize(os);
}

ncvslideio::MediaFrame::View::View(Ptrs&& ptrs, Strides&& strs, Callback &&cb)
    : ptr   (std::move(ptrs))
    , stride(std::move(strs))
    , m_cb  (std::move(cb)) {
}

ncvslideio::MediaFrame::View::~View() {
    if (m_cb) {
        m_cb();
    }
}

ncvslideio::util::any ncvslideio::MediaFrame::IAdapter::blobParams() const
{
    // Does nothing by default
    return {};
}

ncvslideio::MediaFrame::IAdapter::~IAdapter() {
}
