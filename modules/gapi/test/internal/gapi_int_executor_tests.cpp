// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2022 Intel Corporation


#include "../test_precomp.hpp"
#include "../gapi_mock_kernels.hpp"

#include <opencv2/gapi/core.hpp>

namespace opencv_test
{

namespace
{

class GMockExecutable final: public ncvslideio::gimpl::GIslandExecutable
{
    virtual inline bool canReshape() const override {
        return m_priv->m_can_reshape;
    }

    virtual void reshape(ade::Graph&, const GCompileArgs&) override
    {
        m_priv->m_reshape_counter++;
    }
    virtual void handleNewStream() override {  }
    virtual void run(std::vector<InObj>&&, std::vector<OutObj>&&) override { }
    virtual bool allocatesOutputs() const override
    {
        return true;
    }

    virtual ncvslideio::RMat allocate(const ncvslideio::GMatDesc&) const override
    {
        m_priv->m_allocate_counter++;
        return ncvslideio::RMat();
    }

    // NB: GMockBackendImpl creates new unique_ptr<GMockExecutable>
    // on every compile call. Need to share counters between instances in order
    // to validate it in tests.
    struct Priv
    {
        bool m_can_reshape;
        int  m_reshape_counter;
        int  m_allocate_counter;
    };

    std::shared_ptr<Priv> m_priv;

public:
    GMockExecutable(bool can_reshape = true)
        : m_priv(new Priv{can_reshape, 0, 0})
    {
    }

    void setReshape(bool can_reshape) { m_priv->m_can_reshape = can_reshape; }

    int getReshapeCounter()  const { return m_priv->m_reshape_counter;  }
    int getAllocateCounter() const { return m_priv->m_allocate_counter; }
};

class GMockBackendImpl final: public ncvslideio::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &,
                              const ade::NodeHandle &,
                              const ncvslideio::GKernelImpl &) override { }

    virtual EPtr compile(const ade::Graph &,
                         const ncvslideio::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &) const override
    {
        ++m_compile_counter;
        return EPtr{new GMockExecutable(m_exec)};
    }

    mutable int     m_compile_counter = 0;
    GMockExecutable m_exec;

    virtual bool controlsMerge() const override {
        return true;
    }

    virtual bool allowsMerge(const ncvslideio::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override {
        return false;
    }

public:
    GMockBackendImpl(const GMockExecutable& exec) : m_exec(exec) { }
    int getCompileCounter() const { return m_compile_counter; }
};

class GMockFunctor : public gapi::cpu::GOCVFunctor
{
public:
    GMockFunctor(ncvslideio::gapi::GBackend backend,
                 const char* id,
                 const Meta &meta,
                 const Impl& impl)
        : gapi::cpu::GOCVFunctor(id, meta, impl), m_backend(backend)
    {
    }

    ncvslideio::gapi::GBackend backend() const override { return m_backend; }

private:
    ncvslideio::gapi::GBackend m_backend;
};

template<typename K, typename Callable>
GMockFunctor mock_kernel(const ncvslideio::gapi::GBackend& backend, Callable c)
{
    using P = ncvslideio::detail::OCVCallHelper<Callable, typename K::InArgs, typename K::OutArgs>;
    return GMockFunctor{ backend
                       , K::id()
                       , &K::getOutMeta
                       , std::bind(&P::callFunctor, std::placeholders::_1, c)
                       };
}

void dummyFooImpl(const ncvslideio::Mat&, ncvslideio::Mat&)                 { }
void dummyBarImpl(const ncvslideio::Mat&, const ncvslideio::Mat&, ncvslideio::Mat&) { }

struct GExecutorReshapeTest: public ::testing::Test
{
    GExecutorReshapeTest()
        : comp([](){
                ncvslideio::GMat in;
                ncvslideio::GMat out = I::Bar::on(I::Foo::on(in), in);
                return ncvslideio::GComputation(in, out);
          })
    {
        backend_impl1 = std::make_shared<GMockBackendImpl>(island1);
        backend1      = ncvslideio::gapi::GBackend{backend_impl1};
        backend_impl2 = std::make_shared<GMockBackendImpl>(island2);
        backend2      = ncvslideio::gapi::GBackend{backend_impl2};
        auto kernel1  = mock_kernel<I::Foo>(backend1, dummyFooImpl);
        auto kernel2  = mock_kernel<I::Bar>(backend2, dummyBarImpl);
        pkg           = ncvslideio::gapi::kernels(kernel1, kernel2);
        in_mat1       = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        in_mat2       = ncvslideio::Mat::eye(64, 64, CV_8UC1);
    }

    ncvslideio::GComputation                  comp;
    GMockExecutable                   island1;
    std::shared_ptr<GMockBackendImpl> backend_impl1;
    ncvslideio::gapi::GBackend                backend1;
    GMockExecutable                   island2;
    std::shared_ptr<GMockBackendImpl> backend_impl2;
    ncvslideio::gapi::GBackend                backend2;
    ncvslideio::GKernelPackage                pkg;
    ncvslideio::Mat                           in_mat1, in_mat2, out_mat;
};

} // anonymous namespace

// FIXME: avoid code duplication
// The below graph and config is taken from ComplexIslands test suite
TEST(GExecutor, SmokeTest)
{
    ncvslideio::GMat    in[2];
    ncvslideio::GMat    tmp[4];
    ncvslideio::GScalar scl;
    ncvslideio::GMat    out[2];

    tmp[0] = ncvslideio::gapi::bitwise_not(ncvslideio::gapi::bitwise_not(in[0]));
    tmp[1] = ncvslideio::gapi::boxFilter(in[1], -1, ncvslideio::Size(3,3));
    tmp[2] = tmp[0] + tmp[1]; // FIXME: handle tmp[2] = tmp[0]+tmp[2] typo
    scl    = ncvslideio::gapi::sum(tmp[1]);
    tmp[3] = ncvslideio::gapi::medianBlur(tmp[1], 3);
    out[0] = tmp[2] + scl;
    out[1] = ncvslideio::gapi::boxFilter(tmp[3], -1, ncvslideio::Size(3,3));

    //       isl0                                         #internal1
    //       ...........................                  .........
    // (in1) -> NotNot ->(tmp0) --> Add ---------> (tmp2) --> AddC -------> (out1)
    //       :.....................^...:                  :..^....:
    //                             :                         :
    //                             :                         :
    //      #internal0             :                         :
    //        .....................:.........                :
    // (in2) -> Blur -> (tmp1) ----'--> Sum ----> (scl0) ----'
    //        :..........:..................:                  isl1
    //                   :           ..............................
    //                   `------------> Median -> (tmp3) --> Blur -------> (out2)
    //                               :............................:

    ncvslideio::gapi::island("isl0", ncvslideio::GIn(in[0], tmp[1]),  ncvslideio::GOut(tmp[2]));
    ncvslideio::gapi::island("isl1", ncvslideio::GIn(tmp[1]), ncvslideio::GOut(out[1]));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat in_mat2 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat out_gapi[2];

    // Run G-API:
    ncvslideio::GComputation(ncvslideio::GIn(in[0],   in[1]),    ncvslideio::GOut(out[0],      out[1]))
              .apply(ncvslideio::gin(in_mat1, in_mat2),  ncvslideio::gout(out_gapi[0], out_gapi[1]));

    // Run OpenCV
    ncvslideio::Mat out_ocv[2];
    {
        ncvslideio::Mat    ocv_tmp0;
        ncvslideio::Mat    ocv_tmp1;
        ncvslideio::Mat    ocv_tmp2;
        ncvslideio::Mat    ocv_tmp3;
        ncvslideio::Scalar ocv_scl;

        ocv_tmp0 = in_mat1; // skip !(!)
        ncvslideio::boxFilter(in_mat2, ocv_tmp1, -1, ncvslideio::Size(3,3));
        ocv_tmp2 = ocv_tmp0 + ocv_tmp1;
        ocv_scl  = ncvslideio::sum(ocv_tmp1);
        ncvslideio::medianBlur(ocv_tmp1, ocv_tmp3, 3);
        out_ocv[0] = ocv_tmp2 + ocv_scl;
        ncvslideio::boxFilter(ocv_tmp3, out_ocv[1], -1, ncvslideio::Size(3,3));
    }

    EXPECT_EQ(0, cvtest::norm(out_gapi[0], out_ocv[0], NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_gapi[1], out_ocv[1], NORM_INF));

    // FIXME: check that GIslandModel has more than 1 island (e.g. fusion
    // with breakdown worked)
}

TEST_F(GExecutorReshapeTest, ReshapeInsteadOfRecompile)
{
    // NB: Initial state
    EXPECT_EQ(0, backend_impl1->getCompileCounter());
    EXPECT_EQ(0, backend_impl2->getCompileCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());
    EXPECT_EQ(0, island2.getReshapeCounter());

    // NB: First compilation.
    comp.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(1, backend_impl1->getCompileCounter());
    EXPECT_EQ(1, backend_impl2->getCompileCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());
    EXPECT_EQ(0, island2.getReshapeCounter());

    // NB: GMockBackendImpl implements "reshape" method,
    // so it won't be recompiled if the meta is changed.
    comp.apply(ncvslideio::gin(in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(1, backend_impl1->getCompileCounter());
    EXPECT_EQ(1, backend_impl2->getCompileCounter());
    EXPECT_EQ(1, island1.getReshapeCounter());
    EXPECT_EQ(1, island2.getReshapeCounter());
}

TEST_F(GExecutorReshapeTest, OneBackendNotReshapable)
{
    // NB: Make first island not reshapable
    island1.setReshape(false);

    // NB: Initial state
    EXPECT_EQ(0, backend_impl1->getCompileCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());
    EXPECT_EQ(0, backend_impl2->getCompileCounter());
    EXPECT_EQ(0, island2.getReshapeCounter());

    // NB: First compilation.
    comp.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(1, backend_impl1->getCompileCounter());
    EXPECT_EQ(1, backend_impl2->getCompileCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());
    EXPECT_EQ(0, island2.getReshapeCounter());

    // NB: Since one of islands isn't reshapable
    // the entire graph isn't reshapable as well.
    comp.apply(ncvslideio::gin(in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(2, backend_impl1->getCompileCounter());
    EXPECT_EQ(2, backend_impl2->getCompileCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());
    EXPECT_EQ(0, island2.getReshapeCounter());
}

TEST_F(GExecutorReshapeTest, ReshapeCallAllocate)
{
    // NB: Initial state
    EXPECT_EQ(0, island1.getAllocateCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());

    // NB: First compilation.
    comp.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(1, island1.getAllocateCounter());
    EXPECT_EQ(0, island1.getReshapeCounter());

    // NB: The entire graph is reshapable, so it won't be recompiled, but reshaped.
    // Check that reshape call "allocate" to reallocate buffers.
    comp.apply(ncvslideio::gin(in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));
    EXPECT_EQ(2, island1.getAllocateCounter());
    EXPECT_EQ(1, island1.getReshapeCounter());
}

TEST_F(GExecutorReshapeTest, CPUBackendIsReshapable)
{
    comp = ncvslideio::GComputation([](){
        ncvslideio::GMat in;
        ncvslideio::GMat foo = I::Foo::on(in);
        ncvslideio::GMat out = ncvslideio::gapi::bitwise_not(ncvslideio::gapi::bitwise_not(in));
        return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(foo, out));
    });
    // NB: Initial state
    EXPECT_EQ(0, island1.getReshapeCounter());

    // NB: First compilation.
    ncvslideio::Mat out_mat2;
    comp.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat, out_mat2), ncvslideio::compile_args(pkg));
    EXPECT_EQ(0, island1.getReshapeCounter());

    // NB: The entire graph is reshapable, so it won't be recompiled, but reshaped.
    comp.apply(ncvslideio::gin(in_mat2), ncvslideio::gout(out_mat, out_mat2), ncvslideio::compile_args(pkg));
    EXPECT_EQ(1, island1.getReshapeCounter());
    EXPECT_EQ(0, cvtest::norm(out_mat2, in_mat2, NORM_INF));
}

// FIXME: Add explicit tests on GMat/GScalar/GArray<T> being connectors
// between executed islands

} // namespace opencv_test
