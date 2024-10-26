#ifndef OPENCV_GAPI_COMMON_SERIALIZATION_HPP
#define OPENCV_GAPI_COMMON_SERIALIZATION_HPP

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <ade/util/iota_range.hpp> // used in the vector<</>>

#include "compiler/gmodel.hpp"
#include "opencv2/gapi/render/render_types.hpp"
#include "opencv2/gapi/s11n.hpp" // basic interfaces

#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4702)
#endif

namespace ncvslideio {
namespace gapi {
namespace s11n {

struct GSerialized {
    std::vector<ncvslideio::gimpl::Op> m_ops;
    std::vector<ncvslideio::gimpl::Data> m_datas;
    ncvslideio::gimpl::DataObjectCounter m_counter;
    ncvslideio::gimpl::Protocol m_proto;

    using data_tag_t = uint64_t;
    std::map<data_tag_t, ncvslideio::gimpl::ConstValue> m_const_datas;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// S11N operators
// Note: operators for basic types are defined in IIStream/IOStream

// G-API types /////////////////////////////////////////////////////////////////

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GCompileArg& arg);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, ncvslideio::util::monostate  );
GAPI_EXPORTS IIStream& operator>> (IIStream& is, ncvslideio::util::monostate &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, ncvslideio::GShape  shape);
GAPI_EXPORTS IIStream& operator>> (IIStream& is, ncvslideio::GShape &shape);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, ncvslideio::detail::ArgKind  k);
GAPI_EXPORTS IIStream& operator>> (IIStream& is, ncvslideio::detail::ArgKind &k);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, ncvslideio::detail::OpaqueKind  k);
GAPI_EXPORTS IIStream& operator>> (IIStream& is, ncvslideio::detail::OpaqueKind &k);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, ncvslideio::gimpl::Data::Storage  s);
GAPI_EXPORTS IIStream& operator>> (IIStream& is, ncvslideio::gimpl::Data::Storage &s);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::DataObjectCounter &c);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::DataObjectCounter &c);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Protocol &p);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::Protocol &p);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GArg &arg);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GArg &arg);

//Forward declaration
//IOStream& operator<< (IOStream& os, const ncvslideio::GRunArg &arg);
//IIStream& operator>> (IIStream& is, ncvslideio::GRunArg &arg);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GKernel &k);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GKernel &k);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GMatDesc &d);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GMatDesc &d);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GScalarDesc &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GScalarDesc &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GOpaqueDesc &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GOpaqueDesc &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GArrayDesc &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GArrayDesc &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::GFrameDesc &);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::GFrameDesc &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::RcDesc &rc);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::RcDesc &rc);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Op &op);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::Op &op);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Data &op);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::Data &op);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::ConstValue &cd);
GAPI_EXPORTS IIStream& operator>> (IIStream& os, ncvslideio::gimpl::ConstValue &cd);

// Render types ////////////////////////////////////////////////////////////////

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Text &t);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Text &t);

GAPI_EXPORTS IOStream& operator<< (IOStream&, const ncvslideio::gapi::wip::draw::FText &);
GAPI_EXPORTS IIStream& operator>> (IIStream&,       ncvslideio::gapi::wip::draw::FText &);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Circle &c);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Circle &c);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Rect &r);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Rect &r);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Image &i);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Image &i);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Mosaic &m);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Mosaic &m);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Poly &p);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Poly &p);

GAPI_EXPORTS IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Line &l);
GAPI_EXPORTS IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Line &l);

// The top-level serialization routine.
// Note it is just a single function which takes a GModel and a list of nodes
// and writes the data to the stream (recursively)
GAPI_EXPORTS void serialize( IOStream& os
                           , const ade::Graph &g
                           , const std::vector<ade::NodeHandle> &nodes);

// The top-level serialization routine.
// Note it is just a single function which takes a GModel and a list of nodes
// and writes the data to the stream (recursively)
GAPI_EXPORTS void serialize( IOStream& os
                           , const ade::Graph &g
                           , const ncvslideio::gimpl::Protocol &p
                           , const std::vector<ade::NodeHandle> &nodes);

// The top-level deserialization routineS.
// Unfortunately the deserialization is a two-step process:
// 1. First we decode a stream into some intermediate representation
//     (called "GSerialized");
// 2. Then we produce an ade::Graph from this intermediate representation.
//
// An ade::Graph can't be produced from the stream immediately
// since every GCompiled object has its own unique ade::Graph, so
// we can't do it once and for all since every compilation process
// is individual and _is_ altering the ade::Graph state (structure and metadata).
// At the same time, we can't hold the reference to "is" within the GComputation
// forever since this input stream may be associated with an external resource
// and have side effects.
//
// Summarizing, the `deserialize()` happens *once per GComputation* immediately
// during the ncvslideio::gapi::deserialize<GComputation>(), and `reconstruct()` happens
// on every compilation process issued for this GComputation.
GAPI_EXPORTS GSerialized deserialize(IIStream& is);
GAPI_EXPORTS void reconstruct(const GSerialized &s, ade::Graph &g);

// FIXME: Basic Stream implementations /////////////////////////////////////////

// Basic in-memory stream implementations.
class GAPI_EXPORTS ByteMemoryOutStream final: public IOStream {
    std::vector<char> m_storage;

    //virtual IOStream& operator << (uint32_t) override;
    //virtual IOStream& operator<< (uint32_t) final;
public:
    const std::vector<char>& data() const;

    virtual IOStream& operator<< (bool) override;
    virtual IOStream& operator<< (char) override;
    virtual IOStream& operator<< (unsigned char) override;
    virtual IOStream& operator<< (short) override;
    virtual IOStream& operator<< (unsigned short) override;
    virtual IOStream& operator<< (int) override;
    //virtual IOStream& operator<< (std::size_t) override;
    virtual IOStream& operator<< (float) override;
    virtual IOStream& operator<< (double) override;
    virtual IOStream& operator<< (const std::string&) override;
    virtual IOStream& operator<< (uint32_t) override;
    virtual IOStream& operator<< (uint64_t) override;
};

class GAPI_EXPORTS ByteMemoryInStream final: public IIStream {
    const std::vector<char>& m_storage;
    size_t m_idx = 0u;

    void check(std::size_t n) { (void) n; GAPI_DbgAssert(m_idx+n-1 < m_storage.size()); }
    uint32_t getU32() { uint32_t v{}; *this >> v; return v; }

    //virtual IIStream& operator>> (uint32_t &) final;

public:
    explicit ByteMemoryInStream(const std::vector<char> &data);

    virtual IIStream& operator>> (bool &) override;
    virtual IIStream& operator>> (std::vector<bool>::reference) override;
    virtual IIStream& operator>> (char &) override;
    virtual IIStream& operator>> (unsigned char &) override;
    virtual IIStream& operator>> (short &) override;
    virtual IIStream& operator>> (unsigned short &) override;
    virtual IIStream& operator>> (int &) override;
    virtual IIStream& operator>> (float &) override;
    virtual IIStream& operator>> (double &) override;
    //virtual IIStream& operator>> (std::size_t &) override;
    virtual IIStream& operator >> (uint32_t &) override;
    virtual IIStream& operator >> (uint64_t &) override;
    virtual IIStream& operator>> (std::string &) override;
};

namespace detail {
GAPI_EXPORTS std::unique_ptr<IIStream> getInStream(const std::vector<char> &p);
} // namespace detail

GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GCompileArgs &ca);
GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GMetaArgs &ma);
GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GRunArgs &ra);
GAPI_EXPORTS void serialize(IOStream& os, const std::vector<std::string> &vs);
GAPI_EXPORTS GMetaArgs meta_args_deserialize(IIStream& is);
GAPI_EXPORTS GRunArgs run_args_deserialize(IIStream& is);
GAPI_EXPORTS std::vector<std::string> vector_of_strings_deserialize(IIStream& is);

} // namespace s11n
} // namespace gapi
} // namespace ncvslideio

#if defined _MSC_VER
#pragma warning(pop)
#endif

#endif // OPENCV_GAPI_COMMON_SERIALIZATION_HPP
