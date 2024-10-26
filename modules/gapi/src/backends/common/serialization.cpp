// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include <set> // set
#include <map> // map
#include <ade/util/zip_range.hpp> // indexed

#ifdef _WIN32
#define NOMINMAX
#include <winsock.h>      // htonl, ntohl
#else
#include <netinet/in.h>   // htonl, ntohl
#endif

#include <opencv2/gapi/gtype_traits.hpp>

#include "backends/common/serialization.hpp"

namespace ncvslideio {
namespace gapi {
namespace s11n {
namespace {

void putData(GSerialized& s, const ncvslideio::gimpl::GModel::ConstGraph& cg, const ade::NodeHandle &nh) {
    const auto gdata = cg.metadata(nh).get<gimpl::Data>();
    const auto it = ade::util::find_if(s.m_datas, [&gdata](const ncvslideio::gimpl::Data &cd) {
            return cd.rc == gdata.rc && cd.shape == gdata.shape;
        });
    if (s.m_datas.end() == it) {
        s.m_datas.push_back(gdata);

        if (cg.metadata(nh).contains<gimpl::ConstValue>()) {
            size_t datas_num = s.m_datas.size() - 1;
            GAPI_DbgAssert(datas_num <= static_cast<size_t>(std::numeric_limits<GSerialized::data_tag_t>::max()));
            GSerialized::data_tag_t tag = static_cast<GSerialized::data_tag_t>(datas_num);
            s.m_const_datas.emplace(tag,
                                    cg.metadata(nh).get<gimpl::ConstValue>());
        }
    }
}

void putOp(GSerialized& s, const ncvslideio::gimpl::GModel::ConstGraph& cg, const ade::NodeHandle &nh) {
    const auto& op = cg.metadata(nh).get<gimpl::Op>();
    for (const auto &in_nh  : nh->inNodes())  { putData(s, cg, in_nh);  }
    for (const auto &out_nh : nh->outNodes()) { putData(s, cg, out_nh); }
    s.m_ops.push_back(op);
}

ade::NodeHandle mkDataNode(ade::Graph& g, const ncvslideio::gimpl::Data& data) {
    ncvslideio::gimpl::GModel::Graph gm(g);
    auto nh = gm.createNode();
    gm.metadata(nh).set(ncvslideio::gimpl::NodeType{ncvslideio::gimpl::NodeType::DATA});
    gm.metadata(nh).set(data);
    return nh;
}

ade::NodeHandle mkConstDataNode(ade::Graph& g, const ncvslideio::gimpl::Data& data, const ncvslideio::gimpl::ConstValue& const_data) {
    auto nh = mkDataNode(g, data);

    ncvslideio::gimpl::GModel::Graph gm(g);
    gm.metadata(nh).set(const_data);
    return nh;
}

void mkOpNode(ade::Graph& g, const ncvslideio::gimpl::Op& op) {
    ncvslideio::gimpl::GModel::Graph gm(g);
    auto nh = gm.createNode();
    gm.metadata(nh).set(ncvslideio::gimpl::NodeType{ncvslideio::gimpl::NodeType::OP});
    gm.metadata(nh).set(op);
}

void linkNodes(ade::Graph& g) {
    std::map<ncvslideio::gimpl::RcDesc, ade::NodeHandle> dataNodes;
    ncvslideio::gimpl::GModel::Graph gm(g);

    for (const auto& nh : g.nodes()) {
        if (gm.metadata(nh).get<ncvslideio::gimpl::NodeType>().t == ncvslideio::gimpl::NodeType::DATA) {
            const auto &d = gm.metadata(nh).get<gimpl::Data>();
            const auto rc = ncvslideio::gimpl::RcDesc{d.rc, d.shape, d.ctor};
            dataNodes[rc] = nh;
        }
    }

    for (const auto& nh : g.nodes()) {
        if (gm.metadata(nh).get<ncvslideio::gimpl::NodeType>().t == ncvslideio::gimpl::NodeType::OP) {
            const auto& op = gm.metadata(nh).get<gimpl::Op>();
            for (const auto in : ade::util::indexed(op.args)) {
                const auto& arg = ade::util::value(in);
                if (arg.kind == ncvslideio::detail::ArgKind::GOBJREF) {
                    const auto idx = ade::util::index(in);
                    const auto rc  = arg.get<gimpl::RcDesc>();
                    const auto& in_nh = dataNodes.at(rc);
                    const auto& in_eh = g.link(in_nh, nh);
                    gm.metadata(in_eh).set(ncvslideio::gimpl::Input{idx});
                }
            }

            for (const auto out : ade::util::indexed(op.outs)) {
                const auto  idx = ade::util::index(out);
                const auto& rc  = ade::util::value(out);
                const auto& out_nh = dataNodes.at(rc);
                const auto& out_eh = g.link(nh, out_nh);
                gm.metadata(out_eh).set(ncvslideio::gimpl::Output{idx});
            }
        }
    }
}

void relinkProto(ade::Graph& g) {
    using namespace ncvslideio::gimpl;
    // identify which node handles map to the protocol
    // input/output object in the reconstructed graph
    using S = std::set<RcDesc>;                  // FIXME: use ...
    using M = std::map<RcDesc, ade::NodeHandle>; // FIXME: unordered!

    GModel::Graph gm(g);
    auto &proto = gm.metadata().get<Protocol>();

    const S set_in(proto.inputs.begin(), proto.inputs.end());
    const S set_out(proto.outputs.begin(), proto.outputs.end());
    M map_in, map_out;

    // Associate the protocol node handles with their resource identifiers
    for (auto &&nh : gm.nodes()) {
        if (gm.metadata(nh).get<NodeType>().t == NodeType::DATA) {
            const auto &d = gm.metadata(nh).get<Data>();
            const auto rc = RcDesc{d.rc, d.shape, d.ctor};
            if (set_in.count(rc) > 0) {
                GAPI_DbgAssert(set_out.count(rc) == 0);
                map_in[rc] = nh;
            } else if (set_out.count(rc) > 0) {
                GAPI_DbgAssert(set_in.count(rc) == 0);
                map_out[rc] = nh;
            }
        }
    }

    // Reconstruct the protocol vectors, ordered
    proto.in_nhs.reserve(proto.inputs.size());
    proto.in_nhs.clear();
    proto.out_nhs.reserve(proto.outputs.size());
    proto.out_nhs.clear();
    for (auto &rc : proto.inputs)  { proto.in_nhs .push_back(map_in .at(rc)); }
    for (auto &rc : proto.outputs) { proto.out_nhs.push_back(map_out.at(rc)); }

    // If a subgraph is being serialized it's possible that
    // some of its in/out nodes are INTERNAL in the full graph.
    // Set their storage apporpriately
    for (auto &nh : proto.in_nhs)  { gm.metadata(nh).get<Data>().storage = Data::Storage::INPUT; }
    for (auto &nh : proto.out_nhs) { gm.metadata(nh).get<Data>().storage = Data::Storage::OUTPUT; }
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Graph dump operators

// OpenCV types ////////////////////////////////////////////////////////////////

IOStream& operator<< (IOStream& os, const ncvslideio::Point &pt) {
    return os << pt.x << pt.y;
}
IIStream& operator>> (IIStream& is, ncvslideio::Point& pt) {
    return is >> pt.x >> pt.y;
}

IOStream& operator<< (IOStream& os, const ncvslideio::Point2f &pt) {
    return os << pt.x << pt.y;
}
IIStream& operator>> (IIStream& is, ncvslideio::Point2f& pt) {
    return is >> pt.x >> pt.y;
}

IOStream& operator<< (IOStream& os, const ncvslideio::Point3f &pt) {
    return os << pt.x << pt.y << pt.z;
}
IIStream& operator>> (IIStream& is, ncvslideio::Point3f& pt) {
    return is >> pt.x >> pt.y >> pt.z;
}

IOStream& operator<< (IOStream& os, const ncvslideio::Size &sz) {
    return os << sz.width << sz.height;
}
IIStream& operator>> (IIStream& is, ncvslideio::Size& sz) {
    return is >> sz.width >> sz.height;
}

IOStream& operator<< (IOStream& os, const ncvslideio::Rect &rc) {
    return os << rc.x << rc.y << rc.width << rc.height;
}
IIStream& operator>> (IIStream& is, ncvslideio::Rect& rc) {
    return is >> rc.x >> rc.y >> rc.width >> rc.height;
}

IOStream& operator<< (IOStream& os, const ncvslideio::Scalar &s) {
    return os << s.val[0] << s.val[1] << s.val[2] << s.val[3];
}
IIStream& operator>> (IIStream& is, ncvslideio::Scalar& s) {
    return is >> s.val[0] >> s.val[1] >> s.val[2] >> s.val[3];
}
IOStream& operator<< (IOStream& os, const ncvslideio::RMat& mat) {
    mat.serialize(os);
    return os;
}
IIStream& operator>> (IIStream& is, ncvslideio::RMat&) {
    util::throw_error(std::logic_error("operator>> for RMat should never be called. "
                                        "Instead, ncvslideio::gapi::deserialize<ncvslideio::GRunArgs, AdapterTypes...>() "
                                        "should be used"));
    return is;
}

IOStream& operator<< (IOStream& os, const ncvslideio::MediaFrame &frame) {
    frame.serialize(os);
    return os;
}
IIStream& operator>> (IIStream& is, ncvslideio::MediaFrame &) {
    util::throw_error(std::logic_error("operator>> for MediaFrame should never be called. "
                                        "Instead, ncvslideio::gapi::deserialize<ncvslideio::GRunArgs, AdapterTypes...>() "
                                        "should be used"));
    return is;
}

namespace
{

#if !defined(GAPI_STANDALONE)
template<typename T>
    void write_plain(IOStream &os, const T *arr, std::size_t sz) {
        for (auto &&it : ade::util::iota(sz)) os << arr[it];
}
template<typename T>
    void read_plain(IIStream &is, T *arr, std::size_t sz) {
        for (auto &&it : ade::util::iota(sz)) is >> arr[it];
}
template<typename T>
void write_mat_data(IOStream &os, const ncvslideio::Mat &m) {
    // Write every row individually (handles the case when Mat is a view)
    for (auto &&r : ade::util::iota(m.rows)) {
        write_plain(os, m.ptr<T>(r), m.cols*m.channels());
    }
}
template<typename T>
void read_mat_data(IIStream &is, ncvslideio::Mat &m) {
    // Write every row individually (handles the case when Mat is aligned)
    for (auto &&r : ade::util::iota(m.rows)) {
        read_plain(is, m.ptr<T>(r), m.cols*m.channels());
    }
}
#else
void write_plain(IOStream &os, const uchar *arr, std::size_t sz) {
    for (auto &&it : ade::util::iota(sz)) os << arr[it];
}
void read_plain(IIStream &is, uchar *arr, std::size_t sz) {
    for (auto &&it : ade::util::iota(sz)) is >> arr[it];
}
template<typename T>
void write_mat_data(IOStream &os, const ncvslideio::Mat &m) {
    // Write every row individually (handles the case when Mat is a view)
    for (auto &&r : ade::util::iota(m.rows)) {
        write_plain(os, m.ptr(r), m.cols*m.channels()*sizeof(T));
    }
}
template<typename T>
void read_mat_data(IIStream &is, ncvslideio::Mat &m) {
    // Write every row individually (handles the case when Mat is aligned)
    for (auto &&r : ade::util::iota(m.rows)) {
        read_plain(is, m.ptr(r), m.cols*m.channels()*sizeof(T));
    }
}
#endif
} // namespace

IOStream& operator<< (IOStream& os, const ncvslideio::Mat &m) {
#if !defined(GAPI_STANDALONE)
    GAPI_Assert(m.size.dims() == 2 && "Only 2D images are supported now");
#else
    GAPI_Assert(m.dims.size() == 2 && "Only 2D images are supported now");
#endif
    os << m.rows << m.cols << m.type();
    switch (m.depth()) {
    case CV_8U:  write_mat_data< uint8_t>(os, m); break;
    case CV_8S:  write_mat_data<    char>(os, m); break;
    case CV_16U: write_mat_data<uint16_t>(os, m); break;
    case CV_16S: write_mat_data< int16_t>(os, m); break;
    case CV_32S: write_mat_data< int32_t>(os, m); break;
    case CV_32F: write_mat_data<   float>(os, m); break;
    case CV_64F: write_mat_data<  double>(os, m); break;
    default: GAPI_Error("Unsupported Mat depth");
    }
    return os;
}
IIStream& operator>> (IIStream& is, ncvslideio::Mat& m) {
    int rows = -1, cols = -1, type = 0;
    is >> rows >> cols >> type;
    m.create(ncvslideio::Size(cols, rows), type);
    switch (m.depth()) {
    case CV_8U:  read_mat_data< uint8_t>(is, m); break;
    case CV_8S:  read_mat_data<    char>(is, m); break;
    case CV_16U: read_mat_data<uint16_t>(is, m); break;
    case CV_16S: read_mat_data< int16_t>(is, m); break;
    case CV_32S: read_mat_data< int32_t>(is, m); break;
    case CV_32F: read_mat_data<   float>(is, m); break;
    case CV_64F: read_mat_data<  double>(is, m); break;
    default: GAPI_Error("Unsupported Mat depth");
    }
    return is;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Text &t) {
    return os << t.bottom_left_origin << t.color << t.ff << t.fs << t.lt << t.org << t.text << t.thick;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Text &t) {
    return is >> t.bottom_left_origin >> t.color >> t.ff >> t.fs >> t.lt >> t.org >> t.text >> t.thick;
}

IOStream& operator<< (IOStream&, const ncvslideio::gapi::wip::draw::FText &) {
    GAPI_Error("Serialization: Unsupported << for FText");
}
IIStream& operator>> (IIStream&,       ncvslideio::gapi::wip::draw::FText &) {
    GAPI_Error("Serialization: Unsupported >> for FText");
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Circle &c) {
    return os << c.center << c.color << c.lt << c.radius << c.shift << c.thick;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Circle &c) {
    return is >> c.center >> c.color >> c.lt >> c.radius >> c.shift >> c.thick;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Rect &r) {
    return os << r.color << r.lt << r.rect << r.shift << r.thick;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Rect &r) {
    return is >> r.color >> r.lt >> r.rect >> r.shift >> r.thick;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Image &i) {
    return os << i.org << i.alpha << i.img;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Image &i) {
    return is >> i.org >> i.alpha >> i.img;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Mosaic &m) {
    return os << m.cellSz << m.decim << m.mos;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Mosaic &m) {
    return is >> m.cellSz >> m.decim >> m.mos;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Poly &p) {
    return os << p.color << p.lt << p.points << p.shift << p.thick;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Poly &p) {
    return is >> p.color >> p.lt >> p.points >> p.shift >> p.thick;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::draw::Line &l) {
    return os << l.color << l.lt << l.pt1 << l.pt2 << l.shift << l.thick;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gapi::wip::draw::Line &l) {
    return is >> l.color >> l.lt >> l.pt1 >> l.pt2 >> l.shift >> l.thick;
}

// G-API types /////////////////////////////////////////////////////////////////

IOStream& operator<< (IOStream& os, const ncvslideio::GCompileArg& arg)
{
    ByteMemoryOutStream tmpS;
    arg.serialize(tmpS);
    std::vector<char> data = tmpS.data();

    os << arg.tag;
    os << data;

    return os;
}

// Stubs (empty types)

IOStream& operator<< (IOStream& os, ncvslideio::util::monostate  ) {return os;}
IIStream& operator>> (IIStream& is, ncvslideio::util::monostate &) {return is;}

IOStream& operator<< (IOStream& os, const ncvslideio::GScalarDesc &) {return os;}
IIStream& operator>> (IIStream& is,       ncvslideio::GScalarDesc &) {return is;}

IOStream& operator<< (IOStream& os, const ncvslideio::GOpaqueDesc &) {return os;}
IIStream& operator>> (IIStream& is,       ncvslideio::GOpaqueDesc &) {return is;}

IOStream& operator<< (IOStream& os, const ncvslideio::GArrayDesc &) {return os;}
IIStream& operator>> (IIStream& is,       ncvslideio::GArrayDesc &) {return is;}

#if !defined(GAPI_STANDALONE)
IOStream& operator<< (IOStream& os, const ncvslideio::UMat &)
{
    GAPI_Error("Serialization: Unsupported << for UMat");
    return os;
}
IIStream& operator >> (IIStream& is, ncvslideio::UMat &)
{
    GAPI_Error("Serialization: Unsupported >> for UMat");
    return is;
}
#endif // !defined(GAPI_STANDALONE)

IOStream& operator<< (IOStream& os, const ncvslideio::gapi::wip::IStreamSource::Ptr &)
{
    GAPI_Error("Serialization: Unsupported << for IStreamSource::Ptr");
    return os;
}
IIStream& operator >> (IIStream& is, ncvslideio::gapi::wip::IStreamSource::Ptr &)
{
    GAPI_Assert("Serialization: Unsupported >> for IStreamSource::Ptr");
    return is;
}

namespace
{
template<typename Ref, typename T, typename... Ts>
struct putToStream;

template<typename Ref>
struct putToStream<Ref, std::tuple<>>
{
    static void put(IOStream&, const Ref &)
    {
        GAPI_Error("Unsupported type for GArray/GOpaque serialization");
    }
};

template<typename Ref, typename T, typename... Ts>
struct putToStream<Ref, std::tuple<T, Ts...>>
{
    static void put(IOStream& os, const Ref &r)
    {
        if (r.getKind() == ncvslideio::detail::GOpaqueTraits<T>::kind) {
            os << r.template rref<T>();
        } else {
            putToStream<Ref, std::tuple<Ts...> >::put(os, r);
        }
    }
};

template<typename Ref, typename T, typename... Ts>
struct getFromStream;

template<typename Ref>
struct getFromStream<Ref, std::tuple<>>
{
    static void get(IIStream&, Ref &, ncvslideio::detail::OpaqueKind)
    {
        GAPI_Error("Unsupported type for GArray/GOpaque deserialization");
    }
};

template<typename Ref, typename T, typename... Ts>
struct getFromStream<Ref, std::tuple<T, Ts...>>
{
    static void get(IIStream& is, Ref &r, ncvslideio::detail::OpaqueKind kind) {
        if (kind == ncvslideio::detail::GOpaqueTraits<T>::kind) {
            r.template reset<T>();
            auto& val = r.template wref<T>();
            is >> val;
        } else {
            getFromStream<Ref, std::tuple<Ts...> >::get(is, r, kind);
        }
    }
};
}

IOStream& operator<< (IOStream& os, const ncvslideio::detail::VectorRef& ref)
{
    os << ref.getKind();
    putToStream<ncvslideio::detail::VectorRef, ncvslideio::detail::GOpaqueTraitsArrayTypes>::put(os, ref);
    return os;
}
IIStream& operator >> (IIStream& is, ncvslideio::detail::VectorRef& ref)
{
    ncvslideio::detail::OpaqueKind kind;
    is >> kind;
    getFromStream<ncvslideio::detail::VectorRef, ncvslideio::detail::GOpaqueTraitsArrayTypes>::get(is, ref, kind);
    return is;
}

IOStream& operator<< (IOStream& os, const ncvslideio::detail::OpaqueRef& ref)
{
    os << ref.getKind();
    putToStream<ncvslideio::detail::OpaqueRef, ncvslideio::detail::GOpaqueTraitsOpaqueTypes>::put(os, ref);
    return os;
}
IIStream& operator >> (IIStream& is, ncvslideio::detail::OpaqueRef& ref)
{
    ncvslideio::detail::OpaqueKind kind;
    is >> kind;
    getFromStream<ncvslideio::detail::OpaqueRef, ncvslideio::detail::GOpaqueTraitsOpaqueTypes>::get(is, ref, kind);
    return is;
}
// Enums and structures

namespace {
template<typename E> IOStream& put_enum(IOStream& os, E e) {
    return os << static_cast<int>(e);
}
template<typename E> IIStream& get_enum(IIStream& is, E &e) {
    int x{}; is >> x; e = static_cast<E>(x);
    return is;
}
} // anonymous namespace

IOStream& operator<< (IOStream& os, ncvslideio::GShape  sh) {
    return put_enum(os, sh);
}
IIStream& operator>> (IIStream& is, ncvslideio::GShape &sh) {
    return get_enum<ncvslideio::GShape>(is, sh);
}
IOStream& operator<< (IOStream& os, ncvslideio::detail::ArgKind  k) {
    return put_enum(os, k);
}
IIStream& operator>> (IIStream& is, ncvslideio::detail::ArgKind &k) {
    return get_enum<ncvslideio::detail::ArgKind>(is, k);
}
IOStream& operator<< (IOStream& os, ncvslideio::detail::OpaqueKind  k) {
    return put_enum(os, k);
}
IIStream& operator>> (IIStream& is, ncvslideio::detail::OpaqueKind &k) {
    return get_enum<ncvslideio::detail::OpaqueKind>(is, k);
}
IOStream& operator<< (IOStream& os, ncvslideio::gimpl::Data::Storage s) {
    return put_enum(os, s);
}
IIStream& operator>> (IIStream& is, ncvslideio::gimpl::Data::Storage &s) {
    return get_enum<ncvslideio::gimpl::Data::Storage>(is, s);
}

IOStream& operator<< (IOStream& os, const ncvslideio::GArg &arg) {
    // Only GOBJREF and OPAQUE_VAL kinds can be serialized/deserialized
    GAPI_Assert(   arg.kind == ncvslideio::detail::ArgKind::OPAQUE_VAL
                || arg.kind == ncvslideio::detail::ArgKind::GOBJREF);

    os << arg.kind << arg.opaque_kind;
    if (arg.kind == ncvslideio::detail::ArgKind::GOBJREF) {
        os << arg.get<ncvslideio::gimpl::RcDesc>();
    } else {
        GAPI_Assert(arg.kind == ncvslideio::detail::ArgKind::OPAQUE_VAL);
        GAPI_Assert(arg.opaque_kind != ncvslideio::detail::OpaqueKind::CV_UNKNOWN);
        switch (arg.opaque_kind) {
        case ncvslideio::detail::OpaqueKind::CV_BOOL:    os << arg.get<bool>();         break;
        case ncvslideio::detail::OpaqueKind::CV_INT:     os << arg.get<int>();          break;
        case ncvslideio::detail::OpaqueKind::CV_UINT64:  os << arg.get<uint64_t>();     break;
        case ncvslideio::detail::OpaqueKind::CV_DOUBLE:  os << arg.get<double>();       break;
        case ncvslideio::detail::OpaqueKind::CV_FLOAT:   os << arg.get<float>();        break;
        case ncvslideio::detail::OpaqueKind::CV_STRING:  os << arg.get<std::string>();  break;
        case ncvslideio::detail::OpaqueKind::CV_POINT:   os << arg.get<ncvslideio::Point>();    break;
        case ncvslideio::detail::OpaqueKind::CV_SIZE:    os << arg.get<ncvslideio::Size>();     break;
        case ncvslideio::detail::OpaqueKind::CV_RECT:    os << arg.get<ncvslideio::Rect>();     break;
        case ncvslideio::detail::OpaqueKind::CV_SCALAR:  os << arg.get<ncvslideio::Scalar>();   break;
        case ncvslideio::detail::OpaqueKind::CV_MAT:     os << arg.get<ncvslideio::Mat>();      break;
        default: GAPI_Error("GArg: Unsupported (unknown?) opaque value type");
        }
    }
    return os;
}

IIStream& operator>> (IIStream& is, ncvslideio::GArg &arg) {
    is >> arg.kind >> arg.opaque_kind;

    // Only GOBJREF and OPAQUE_VAL kinds can be serialized/deserialized
    GAPI_Assert(   arg.kind == ncvslideio::detail::ArgKind::OPAQUE_VAL
                || arg.kind == ncvslideio::detail::ArgKind::GOBJREF);

    if (arg.kind == ncvslideio::detail::ArgKind::GOBJREF) {
        ncvslideio::gimpl::RcDesc rc;
        is >> rc;
        arg = (GArg(rc));
    } else {
        GAPI_Assert(arg.kind == ncvslideio::detail::ArgKind::OPAQUE_VAL);
        GAPI_Assert(arg.opaque_kind != ncvslideio::detail::OpaqueKind::CV_UNKNOWN);
        switch (arg.opaque_kind) {
#define HANDLE_CASE(E,T) case ncvslideio::detail::OpaqueKind::CV_##E:           \
            { T t{}; is >> t; arg = (ncvslideio::GArg(t)); } break
            HANDLE_CASE(BOOL    , bool);
            HANDLE_CASE(INT     , int);
            HANDLE_CASE(UINT64  , uint64_t);
            HANDLE_CASE(DOUBLE  , double);
            HANDLE_CASE(FLOAT   , float);
            HANDLE_CASE(STRING  , std::string);
            HANDLE_CASE(POINT   , ncvslideio::Point);
            HANDLE_CASE(POINT2F , ncvslideio::Point2f);
            HANDLE_CASE(POINT3F , ncvslideio::Point3f);
            HANDLE_CASE(SIZE    , ncvslideio::Size);
            HANDLE_CASE(RECT    , ncvslideio::Rect);
            HANDLE_CASE(SCALAR  , ncvslideio::Scalar);
            HANDLE_CASE(MAT     , ncvslideio::Mat);
#undef HANDLE_CASE
        default: GAPI_Error("GArg: Unsupported (unknown?) opaque value type");
        }
    }
    return is;
}

IOStream& operator<< (IOStream& os, const ncvslideio::GKernel &k) {
    return os << k.name << k.tag << k.outShapes;
}
IIStream& operator>> (IIStream& is, ncvslideio::GKernel &k) {
    return is >> const_cast<std::string&>(k.name)
              >> const_cast<std::string&>(k.tag)
              >> const_cast<ncvslideio::GShapes&>(k.outShapes);
}


IOStream& operator<< (IOStream& os, const ncvslideio::GMatDesc &d) {
    return os << d.depth << d.chan << d.size << d.planar << d.dims;
}
IIStream& operator>> (IIStream& is, ncvslideio::GMatDesc &d) {
    return is >> d.depth >> d.chan >> d.size >> d.planar >> d.dims;
}

IOStream& operator<< (IOStream& os, const ncvslideio::GFrameDesc &d) {
    return put_enum(os, d.fmt) << d.size;
}
IIStream& operator>> (IIStream& is,       ncvslideio::GFrameDesc &d) {
    return get_enum(is, d.fmt) >> d.size;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::RcDesc &rc) {
    // FIXME: HostCtor is not serialized!
    return os << rc.id << rc.shape;
}
IIStream& operator>> (IIStream& is, ncvslideio::gimpl::RcDesc &rc) {
    // FIXME: HostCtor is not deserialized!
    return is >> rc.id >> rc.shape;
}


IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Op &op) {
    return os << op.k << op.args << op.outs;
}
IIStream& operator>> (IIStream& is, ncvslideio::gimpl::Op &op) {
    return is >> op.k >> op.args >> op.outs;
}


IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Data &d) {
    // FIXME: HostCtor is not stored here!!
    // FIXME: Storage may be incorrect for subgraph-to-graph process
    return os << d.shape << d.rc << d.meta << d.storage << d.kind;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::ConstValue &cd) {
    return os << cd.arg;
}

namespace
{
template<typename Ref, typename T, typename... Ts>
struct initCtor;

template<typename Ref>
struct initCtor<Ref, std::tuple<>>
{
    static void init(ncvslideio::gimpl::Data&)
    {
        GAPI_Error("Unsupported type for GArray/GOpaque deserialization");
    }
};

template<typename Ref, typename T, typename... Ts>
struct initCtor<Ref, std::tuple<T, Ts...>>
{
    static void init(ncvslideio::gimpl::Data& d) {
        if (d.kind == ncvslideio::detail::GOpaqueTraits<T>::kind) {
            static std::function<void(Ref&)> ctor = [](Ref& ref){ref.template reset<T>();};
            d.ctor = ctor;
        } else {
            initCtor<Ref, std::tuple<Ts...> >::init(d);
        }
    }
};
} // anonymous namespace

IIStream& operator>> (IIStream& is, ncvslideio::gimpl::Data &d) {
    // FIXME: HostCtor is not stored here!!
    // FIXME: Storage may be incorrect for subgraph-to-graph process
    is >> d.shape >> d.rc >> d.meta >> d.storage >> d.kind;
    if (d.shape == ncvslideio::GShape::GARRAY)
    {
        initCtor<ncvslideio::detail::VectorRef, ncvslideio::detail::GOpaqueTraitsArrayTypes>::init(d);
    }
    else if (d.shape == ncvslideio::GShape::GOPAQUE)
    {
        initCtor<ncvslideio::detail::OpaqueRef, ncvslideio::detail::GOpaqueTraitsOpaqueTypes>::init(d);
    }
    return is;
}

IIStream& operator>> (IIStream& is, ncvslideio::gimpl::ConstValue &cd) {
    return is >> cd.arg;
}

IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::DataObjectCounter &c) {
    return os << c.m_next_data_id;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::DataObjectCounter &c) {
    return is >> c.m_next_data_id;
}


IOStream& operator<< (IOStream& os, const ncvslideio::gimpl::Protocol &p) {
    // NB: in_nhs/out_nhs are not written!
    return os << p.inputs << p.outputs;
}
IIStream& operator>> (IIStream& is,       ncvslideio::gimpl::Protocol &p) {
    // NB: in_nhs/out_nhs are reconstructed at a later phase
    return is >> p.inputs >> p.outputs;
}


void serialize( IOStream& os
              , const ade::Graph &g
              , const std::vector<ade::NodeHandle> &nodes) {
    ncvslideio::gimpl::GModel::ConstGraph cg(g);
    serialize(os, g, cg.metadata().get<ncvslideio::gimpl::Protocol>(), nodes);
}

void serialize( IOStream& os
              , const ade::Graph &g
              , const ncvslideio::gimpl::Protocol &p
              , const std::vector<ade::NodeHandle> &nodes) {
    ncvslideio::gimpl::GModel::ConstGraph cg(g);
    GSerialized s;
    for (auto &nh : nodes) {
        switch (cg.metadata(nh).get<ncvslideio::gimpl::NodeType>().t)
        {
        case ncvslideio::gimpl::NodeType::OP:   putOp  (s, cg, nh); break;
        case ncvslideio::gimpl::NodeType::DATA: putData(s, cg, nh); break;
        default: util::throw_error(std::logic_error("Unknown NodeType"));
        }
    }
    s.m_counter = cg.metadata().get<ncvslideio::gimpl::DataObjectCounter>();
    s.m_proto   = p;
    os << s.m_ops << s.m_datas << s.m_counter << s.m_proto << s.m_const_datas;
}

GSerialized deserialize(IIStream &is) {
    GSerialized s;
    is >> s.m_ops >> s.m_datas >> s.m_counter >> s.m_proto >> s.m_const_datas;
    return s;
}

void reconstruct(const GSerialized &s, ade::Graph &g) {
    GAPI_Assert(g.nodes().empty());

    GSerialized::data_tag_t tag = 0;
    for (const auto& d  : s.m_datas) {
        if (d.storage == gimpl::Data::Storage::CONST_VAL) {
            auto cit = s.m_const_datas.find(tag);
            if (cit == s.m_const_datas.end()) {
                util::throw_error(std::logic_error("Cannot reconstruct graph: Data::Storage::CONST_VAL by tag: " +
                                  std::to_string(tag) + " requires ConstValue"));
            }

            mkConstDataNode(g, d, cit->second);
        } else {
            ncvslideio::gapi::s11n::mkDataNode(g, d);
        }

        tag ++;
    }
    for (const auto& op : s.m_ops)   ncvslideio::gapi::s11n::mkOpNode(g, op);
    ncvslideio::gapi::s11n::linkNodes(g);

    ncvslideio::gimpl::GModel::Graph gm(g);
    gm.metadata().set(s.m_counter);
    gm.metadata().set(s.m_proto);
    ncvslideio::gapi::s11n::relinkProto(g);
    gm.metadata().set(ncvslideio::gimpl::Deserialized{});
}

////////////////////////////////////////////////////////////////////////////////
// Streams /////////////////////////////////////////////////////////////////////

const std::vector<char>& ByteMemoryOutStream::data() const {
    return m_storage;
}
IOStream& ByteMemoryOutStream::operator<< (uint32_t atom) {
    m_storage.push_back(0xFF & (atom));
    m_storage.push_back(0xFF & (atom >> 8));
    m_storage.push_back(0xFF & (atom >> 16));
    m_storage.push_back(0xFF & (atom >> 24));
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (uint64_t atom) {
    for (int i = 0; i < 8; ++i) {
        m_storage.push_back(0xFF & (atom >> (i * 8)));;
    }
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (bool atom) {
    m_storage.push_back(atom ? 1 : 0);
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (char atom) {
    m_storage.push_back(atom);
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (unsigned char atom) {
    return *this << static_cast<char>(atom);
}
IOStream& ByteMemoryOutStream::operator<< (short atom) {
    static_assert(sizeof(short) == 2, "Expecting sizeof(short) == 2");
    m_storage.push_back(0xFF & (atom));
    m_storage.push_back(0xFF & (atom >> 8));
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (unsigned short atom) {
    return *this << static_cast<short>(atom);
}
IOStream& ByteMemoryOutStream::operator<< (int atom) {
    static_assert(sizeof(int) == 4, "Expecting sizeof(int) == 4");
    return *this << static_cast<uint32_t>(atom);
}
//IOStream& ByteMemoryOutStream::operator<< (std::size_t atom) {
//    // NB: type truncated!
//    return *this << static_cast<uint32_t>(atom);
//}
IOStream& ByteMemoryOutStream::operator<< (float atom) {
    static_assert(sizeof(float) == 4, "Expecting sizeof(float) == 4");
    uint32_t tmp = 0u;
    memcpy(&tmp, &atom, sizeof(float));
    return *this << static_cast<uint32_t>(htonl(tmp));
}
IOStream& ByteMemoryOutStream::operator<< (double atom) {
    static_assert(sizeof(double) == 8, "Expecting sizeof(double) == 8");
    uint32_t tmp[2] = {0u};
    memcpy(tmp, &atom, sizeof(double));
    *this << static_cast<uint32_t>(htonl(tmp[0]));
    *this << static_cast<uint32_t>(htonl(tmp[1]));
    return *this;
}
IOStream& ByteMemoryOutStream::operator<< (const std::string &str) {
    //*this << static_cast<std::size_t>(str.size()); // N.B. Put type explicitly
    *this << static_cast<uint32_t>(str.size()); // N.B. Put type explicitly
    for (auto c : str) *this << c;
    return *this;
}
ByteMemoryInStream::ByteMemoryInStream(const std::vector<char> &data)
    : m_storage(data) {
}
IIStream& ByteMemoryInStream::operator>> (uint32_t &atom) {
    check(sizeof(uint32_t));
    uint8_t x[4];
    x[0] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[1] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[2] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[3] = static_cast<uint8_t>(m_storage[m_idx++]);
    atom = ((x[0]) | (x[1] << 8) | (x[2] << 16) | (x[3] << 24));
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (bool& atom) {
    check(sizeof(char));
    atom = (m_storage[m_idx++] == 0) ? false : true;
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (std::vector<bool>::reference atom) {
    check(sizeof(char));
    atom = (m_storage[m_idx++] == 0) ? false : true;
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (char &atom) {
    check(sizeof(char));
    atom = m_storage[m_idx++];
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (uint64_t &atom) {
    check(sizeof(uint64_t));
    uint8_t x[8];
    atom = 0;
    for (int i = 0; i < 8; ++i) {
        x[i] = static_cast<uint8_t>(m_storage[m_idx++]);
        atom |= (static_cast<uint64_t>(x[i]) << (i * 8));
    }
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (unsigned char &atom) {
    char c{};
    *this >> c;
    atom = static_cast<unsigned char>(c);
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (short &atom) {
    static_assert(sizeof(short) == 2, "Expecting sizeof(short) == 2");
    check(sizeof(short));
    uint8_t x[2];
    x[0] = static_cast<uint8_t>(m_storage[m_idx++]);
    x[1] = static_cast<uint8_t>(m_storage[m_idx++]);
    atom = ((x[0]) | (x[1] << 8));
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (unsigned short &atom) {
    short s{};
    *this >> s;
    atom = static_cast<unsigned short>(s);
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (int& atom) {
    static_assert(sizeof(int) == 4, "Expecting sizeof(int) == 4");
    atom = static_cast<int>(getU32());
    return *this;
}
//IIStream& ByteMemoryInStream::operator>> (std::size_t& atom) {
//    // NB. Type was truncated!
//    atom = static_cast<std::size_t>(getU32());
//    return *this;
//}
IIStream& ByteMemoryInStream::operator>> (float& atom) {
    static_assert(sizeof(float) == 4, "Expecting sizeof(float) == 4");
    uint32_t tmp = ntohl(getU32());
    memcpy(&atom, &tmp, sizeof(float));
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (double& atom) {
    static_assert(sizeof(double) == 8, "Expecting sizeof(double) == 8");
    uint32_t tmp[2] = {ntohl(getU32()), ntohl(getU32())};
    memcpy(&atom, tmp, sizeof(double));
    return *this;
}
IIStream& ByteMemoryInStream::operator>> (std::string& str) {
    //std::size_t sz = 0u;
    uint32_t sz = 0u;
    *this >> sz;
    if (sz == 0u) {
        str.clear();
    } else {
        str.resize(static_cast<std::size_t>(sz));
        for (auto &&i : ade::util::iota(sz)) { *this >> str[i]; }
    }
    return *this;
}

GAPI_EXPORTS std::unique_ptr<IIStream> detail::getInStream(const std::vector<char> &p) {
    return std::unique_ptr<ByteMemoryInStream>(new ByteMemoryInStream(p));
}

GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GCompileArgs &ca) {
    os << ca;
}

GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GMetaArgs &ma) {
    os << ma;
}
GAPI_EXPORTS void serialize(IOStream& os, const ncvslideio::GRunArgs &ra) {
    os << ra;
}
GAPI_EXPORTS void serialize(IOStream& os, const std::vector<std::string> &vs) {
    os << vs;
}
GAPI_EXPORTS GMetaArgs meta_args_deserialize(IIStream& is) {
    GMetaArgs s;
    is >> s;
    return s;
}
GAPI_EXPORTS GRunArgs run_args_deserialize(IIStream& is) {
    GRunArgs s;
    is >> s;
    return s;
}
GAPI_EXPORTS std::vector<std::string> vector_of_strings_deserialize(IIStream& is) {
    std::vector<std::string> s;
    is >> s;
    return s;
}

} // namespace s11n
} // namespace gapi
} // namespace ncvslideio
