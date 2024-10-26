#ifndef OPENCV_TS_PERF_HPP
#define OPENCV_TS_PERF_HPP

#include "opencv2/ts.hpp"

#include "ts_ext.hpp"

#include <functional>

#if !(defined(LOGD) || defined(LOGI) || defined(LOGW) || defined(LOGE))
# if defined(__ANDROID__) && defined(USE_ANDROID_LOGGING)
#  include <android/log.h>

#  define PERF_TESTS_LOG_TAG "OpenCV_perf"
#  define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#  define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#  define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, PERF_TESTS_LOG_TAG, __VA_ARGS__))
#  define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, PERF_TESTS_LOG_TAG, __VA_ARGS__))
# else
#  define LOGD(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#  define LOGI(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#  define LOGW(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
#  define LOGE(_str, ...) do{printf(_str , ## __VA_ARGS__); printf("\n");fflush(stdout);} while(0)
# endif
#endif

// declare major namespaces to avoid errors on unknown namespace
namespace ncvslideio { namespace cuda {} namespace ocl {} }
namespace cvtest { }

namespace perf
{

// Tuple stuff from Google Tests
using testing::get;
using testing::make_tuple;
using testing::tuple;
using testing::tuple_size;
using testing::tuple_element;

class TestBase;

/*****************************************************************************************\
*                Predefined typical frame sizes and typical test parameters               *
\*****************************************************************************************/
const static ncvslideio::Size szQVGA = ncvslideio::Size(320, 240);
const static ncvslideio::Size szVGA = ncvslideio::Size(640, 480);
const static ncvslideio::Size szSVGA = ncvslideio::Size(800, 600);
const static ncvslideio::Size szXGA = ncvslideio::Size(1024, 768);
const static ncvslideio::Size szSXGA = ncvslideio::Size(1280, 1024);
const static ncvslideio::Size szWQHD = ncvslideio::Size(2560, 1440);

const static ncvslideio::Size sznHD = ncvslideio::Size(640, 360);
const static ncvslideio::Size szqHD = ncvslideio::Size(960, 540);
const static ncvslideio::Size sz240p = szQVGA;
const static ncvslideio::Size sz720p = ncvslideio::Size(1280, 720);
const static ncvslideio::Size sz1080p = ncvslideio::Size(1920, 1080);
const static ncvslideio::Size sz1440p = szWQHD;
const static ncvslideio::Size sz2160p = ncvslideio::Size(3840, 2160);//UHDTV1 4K
const static ncvslideio::Size sz4320p = ncvslideio::Size(7680, 4320);//UHDTV2 8K

const static ncvslideio::Size sz3MP = ncvslideio::Size(2048, 1536);
const static ncvslideio::Size sz5MP = ncvslideio::Size(2592, 1944);
const static ncvslideio::Size sz2K = ncvslideio::Size(2048, 2048);

const static ncvslideio::Size szODD = ncvslideio::Size(127, 61);

const static ncvslideio::Size szSmall24 = ncvslideio::Size(24, 24);
const static ncvslideio::Size szSmall32 = ncvslideio::Size(32, 32);
const static ncvslideio::Size szSmall64 = ncvslideio::Size(64, 64);
const static ncvslideio::Size szSmall128 = ncvslideio::Size(128, 128);

#define SZ_ALL_VGA ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA)
#define SZ_ALL_GA  ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA, ::perf::szXGA, ::perf::szSXGA)
#define SZ_ALL_HD  ::testing::Values(::perf::sznHD, ::perf::szqHD, ::perf::sz720p, ::perf::sz1080p)
#define SZ_ALL_SMALL ::testing::Values(::perf::szSmall24, ::perf::szSmall32, ::perf::szSmall64, ::perf::szSmall128)
#define SZ_ALL  ::testing::Values(::perf::szQVGA, ::perf::szVGA, ::perf::szSVGA, ::perf::szXGA, ::perf::szSXGA, ::perf::sznHD, ::perf::szqHD, ::perf::sz720p, ::perf::sz1080p)
#define SZ_TYPICAL  ::testing::Values(::perf::szVGA, ::perf::szqHD, ::perf::sz720p, ::perf::szODD)


#define TYPICAL_MAT_SIZES ::perf::szVGA, ::perf::sz720p, ::perf::sz1080p, ::perf::szODD
#define TYPICAL_MAT_TYPES CV_8UC1, CV_8UC4, CV_32FC1
#define TYPICAL_MATS testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( TYPICAL_MAT_TYPES ) )
#define TYPICAL_MATS_C1 testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( CV_8UC1, CV_32FC1 ) )
#define TYPICAL_MATS_C4 testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( CV_8UC4 ) )


/*****************************************************************************************\
*                MatType - printable wrapper over integer 'type' of Mat                   *
\*****************************************************************************************/
class MatType
{
public:
    MatType(int val=0) : _type(val) {}
    operator int() const {return _type;}

private:
    int _type;
};

/*****************************************************************************************\
*     CV_ENUM and CV_FLAGS - macro to create printable wrappers for defines and enums     *
\*****************************************************************************************/

#define CV_ENUM(class_name, ...)                                                        \
    namespace {                                                                         \
    using namespace ncvslideio;using namespace ncvslideio::cuda; using namespace ncvslideio::ocl;               \
    struct class_name {                                                                 \
        class_name(int val = 0) : val_(val) {}                                          \
        operator int() const { return val_; }                                           \
        void PrintTo(std::ostream* os) const {                                          \
            const int vals[] = { __VA_ARGS__ };                                         \
            const char* svals = #__VA_ARGS__;                                           \
            for(int i = 0, pos = 0; i < (int)(sizeof(vals)/sizeof(int)); ++i) {         \
                while(isspace(svals[pos]) || svals[pos] == ',') ++pos;                  \
                int start = pos;                                                        \
                while(!(isspace(svals[pos]) || svals[pos] == ',' || svals[pos] == 0))   \
                    ++pos;                                                              \
                if (val_ == vals[i]) {                                                  \
                    *os << std::string(svals + start, svals + pos);                     \
                    return;                                                             \
                }                                                                       \
            }                                                                           \
            *os << "UNKNOWN";                                                           \
        }                                                                               \
        static ::testing::internal::ParamGenerator<class_name> all() {                  \
            const class_name vals[] = { __VA_ARGS__ };                                  \
            return ::testing::ValuesIn(vals);                                           \
        }                                                                               \
    private: int val_;                                                                  \
    };                                                                                  \
    static inline void PrintTo(const class_name& t, std::ostream* os) { t.PrintTo(os); } }

#define CV_FLAGS(class_name, ...)                                                       \
    namespace {                                                                         \
    struct class_name {                                                                 \
        class_name(int val = 0) : val_(val) {}                                          \
        operator int() const { return val_; }                                           \
        void PrintTo(std::ostream* os) const {                                          \
            using namespace ncvslideio;using namespace ncvslideio::cuda; using namespace ncvslideio::ocl;        \
            const int vals[] = { __VA_ARGS__ };                                         \
            const char* svals = #__VA_ARGS__;                                           \
            int value = val_;                                                           \
            bool first = true;                                                          \
            for(int i = 0, pos = 0; i < (int)(sizeof(vals)/sizeof(int)); ++i) {         \
                while(isspace(svals[pos]) || svals[pos] == ',') ++pos;                  \
                int start = pos;                                                        \
                while(!(isspace(svals[pos]) || svals[pos] == ',' || svals[pos] == 0))   \
                    ++pos;                                                              \
                if ((value & vals[i]) == vals[i]) {                                     \
                    value &= ~vals[i];                                                  \
                    if (first) first = false; else *os << "|";                          \
                    *os << std::string(svals + start, svals + pos);                     \
                    if (!value) return;                                                 \
                }                                                                       \
            }                                                                           \
            if (first) *os << "UNKNOWN";                                                \
        }                                                                               \
    private: int val_;                                                                  \
    };                                                                                  \
    static inline void PrintTo(const class_name& t, std::ostream* os) { t.PrintTo(os); } }

CV_ENUM(MatDepth, CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_16F)

/*****************************************************************************************\
*                 Regression control utility for performance testing                      *
\*****************************************************************************************/
enum ERROR_TYPE
{
    ERROR_ABSOLUTE = 0,
    ERROR_RELATIVE = 1
};

class Regression
{
public:
    static Regression& add(TestBase* test, const std::string& name, ncvslideio::InputArray array, double eps = DBL_EPSILON, ERROR_TYPE err = ERROR_ABSOLUTE);
    static Regression& addMoments(TestBase* test, const std::string& name, const ncvslideio::Moments & array, double eps = DBL_EPSILON, ERROR_TYPE err = ERROR_ABSOLUTE);
    static Regression& addKeypoints(TestBase* test, const std::string& name, const std::vector<ncvslideio::KeyPoint>& array, double eps = DBL_EPSILON, ERROR_TYPE err = ERROR_ABSOLUTE);
    static Regression& addMatches(TestBase* test, const std::string& name, const std::vector<ncvslideio::DMatch>& array, double eps = DBL_EPSILON, ERROR_TYPE err = ERROR_ABSOLUTE);
    static void Init(const std::string& testSuitName, const std::string& ext = ".xml");

    Regression& operator() (const std::string& name, ncvslideio::InputArray array, double eps = DBL_EPSILON, ERROR_TYPE err = ERROR_ABSOLUTE);

private:
    static Regression& instance();
    Regression();
    ~Regression();

    Regression(const Regression&);
    Regression& operator=(const Regression&);

    ncvslideio::RNG regRNG;//own random numbers generator to make collection and verification work identical
    std::string storageInPath;
    std::string storageOutPath;
    ncvslideio::FileStorage storageIn;
    ncvslideio::FileStorage storageOut;
    ncvslideio::FileNode rootIn;
    std::string currentTestNodeName;
    std::string suiteName;

    ncvslideio::FileStorage& write();

    static std::string getCurrentTestNodeName();
    static bool isVector(ncvslideio::InputArray a);
    static double getElem(ncvslideio::Mat& m, int x, int y, int cn = 0);

    void init(const std::string& testSuitName, const std::string& ext);
    void write(ncvslideio::InputArray array);
    void write(ncvslideio::Mat m);
    void verify(ncvslideio::FileNode node, ncvslideio::InputArray array, double eps, ERROR_TYPE err);
    void verify(ncvslideio::FileNode node, ncvslideio::Mat actual, double eps, std::string argname, ERROR_TYPE err);
};

#define SANITY_CHECK(array, ...) ::perf::Regression::add(this, #array, array , ## __VA_ARGS__)
#define SANITY_CHECK_MOMENTS(array, ...) ::perf::Regression::addMoments(this, #array, array , ## __VA_ARGS__)
#define SANITY_CHECK_KEYPOINTS(array, ...) ::perf::Regression::addKeypoints(this, #array, array , ## __VA_ARGS__)
#define SANITY_CHECK_MATCHES(array, ...) ::perf::Regression::addMatches(this, #array, array , ## __VA_ARGS__)
#define SANITY_CHECK_NOTHING() this->setVerified()

class GpuPerf
{
public:
  static bool targetDevice();
};

#define PERF_RUN_CUDA()  ::perf::GpuPerf::targetDevice()

/*****************************************************************************************\
*                            Container for performance metrics                            *
\*****************************************************************************************/
typedef struct performance_metrics
{
    size_t bytesIn;
    size_t bytesOut;
    unsigned int samples;
    unsigned int outliers;
    double gmean;
    double gstddev;//stddev for log(time)
    double mean;
    double stddev;
    double median;
    double min;
    double frequency;
    int terminationReason;

    enum
    {
        TERM_ITERATIONS = 0,
        TERM_TIME = 1,
        TERM_INTERRUPT = 2,
        TERM_EXCEPTION = 3,
        TERM_SKIP_TEST = 4, // there are some limitations and test should be skipped
        TERM_UNKNOWN = -1
    };

    performance_metrics();
    void clear();
} performance_metrics;


/*****************************************************************************************\
*                           Strategy for performance measuring                            *
\*****************************************************************************************/
enum PERF_STRATEGY
{
    PERF_STRATEGY_DEFAULT = -1,
    PERF_STRATEGY_BASE = 0,
    PERF_STRATEGY_SIMPLE = 1
};


/*****************************************************************************************\
*                           Base fixture for performance tests                            *
\*****************************************************************************************/
#ifdef CV_COLLECT_IMPL_DATA
// Implementation collection processing class.
// Accumulates and shapes implementation data.
typedef struct ImplData
{
    bool ipp;
    bool icv;
    bool ipp_mt;
    bool ocl;
    bool plain;
    std::vector<int> implCode;
    std::vector<ncvslideio::String> funName;

    ImplData()
    {
        Reset();
    }

    void Reset()
    {
        ncvslideio::setImpl(0);
        ipp = icv = ocl = ipp_mt = false;
        implCode.clear();
        funName.clear();
    }

    void GetImpl()
    {
        flagsToVars(ncvslideio::getImpl(implCode, funName));
    }

    std::vector<ncvslideio::String> GetCallsForImpl(int impl)
    {
        std::vector<ncvslideio::String> out;

        for(int i = 0; i < (int)implCode.size(); i++)
        {
            if(impl == implCode[i])
                out.push_back(funName[i]);
        }
        return out;
    }

    // Remove duplicate entries
    void ShapeUp()
    {
        std::vector<int> savedCode;
        std::vector<ncvslideio::String> savedName;

        for(int i = 0; i < (int)implCode.size(); i++)
        {
            bool match = false;
            for(int j = 0; j < (int)savedCode.size(); j++)
            {
                if(implCode[i] == savedCode[j] && !funName[i].compare(savedName[j]))
                {
                    match = true;
                    break;
                }
            }
            if(!match)
            {
                savedCode.push_back(implCode[i]);
                savedName.push_back(funName[i]);
            }
        }

        implCode = savedCode;
        funName = savedName;
    }

    // convert flags register to more handy variables
    void flagsToVars(int flags)
    {
#if defined(HAVE_IPP_ICV)
        ipp    = 0;
        icv    = ((flags&CV_IMPL_IPP) > 0);
#else
        ipp    = ((flags&CV_IMPL_IPP) > 0);
        icv    = 0;
#endif
        ipp_mt = ((flags&CV_IMPL_MT) > 0);
        ocl    = ((flags&CV_IMPL_OCL) > 0);
        plain  = (flags == 0);
    }

} ImplData;
#endif

#ifdef ENABLE_INSTRUMENTATION
class InstumentData
{
public:
    static ::ncvslideio::String treeToString();
    static void         printTree();
};
#endif

class TestBase: public ::testing::Test
{
public:
    TestBase();

    static void Init(int argc, const char* const argv[]);
    static void Init(const std::vector<std::string> & availableImpls,
                     int argc, const char* const argv[]);
    static void RecordRunParameters();
    static std::string getDataPath(const std::string& relativePath);
    static std::string getSelectedImpl();

    static enum PERF_STRATEGY getCurrentModulePerformanceStrategy();
    static enum PERF_STRATEGY setModulePerformanceStrategy(enum PERF_STRATEGY strategy);

    class PerfSkipTestException: public cvtest::SkipTestException
    {
    public:
        int dummy; // workaround for MacOSX Xcode 7.3 bug (don't make class "empty")
        PerfSkipTestException() : dummy(0) {}
    };

protected:
    virtual void PerfTestBody() = 0;

    virtual void SetUp() CV_OVERRIDE;
    virtual void TearDown() CV_OVERRIDE;

    bool startTimer(); // bool is dummy for conditional loop
    void stopTimer();
    bool next();

    PERF_STRATEGY getCurrentPerformanceStrategy() const;

    enum WarmUpType
    {
        WARMUP_READ,
        WARMUP_WRITE,
        WARMUP_RNG,
        WARMUP_NONE
    };

    void reportMetrics(bool toJUnitXML = false);
    static void warmup(ncvslideio::InputOutputArray a, WarmUpType wtype = WARMUP_READ);

    performance_metrics& calcMetrics();

    void RunPerfTestBody();

#ifdef CV_COLLECT_IMPL_DATA
    ImplData implConf;
#endif
#ifdef ENABLE_INSTRUMENTATION
    InstumentData instrConf;
#endif

private:
    typedef std::vector<std::pair<int, ncvslideio::Size> > SizeVector;
    typedef std::vector<int64> TimeVector;

    SizeVector inputData;
    SizeVector outputData;
    unsigned int getTotalInputSize() const;
    unsigned int getTotalOutputSize() const;

    enum PERF_STRATEGY testStrategy;

    TimeVector times;
    int64 lastTime;
    int64 totalTime;
    int64 timeLimit;
    static int64 timeLimitDefault;
    static unsigned int iterationsLimitDefault;

    unsigned int minIters;
    unsigned int nIters;
    unsigned int currentIter;
    unsigned int runsPerIteration;
    unsigned int perfValidationStage;

    performance_metrics metrics;
    void validateMetrics();

    static void warmup_impl(ncvslideio::Mat m, WarmUpType wtype);
    static int getSizeInBytes(ncvslideio::InputArray a);
    static ncvslideio::Size getSize(ncvslideio::InputArray a);
    static void declareArray(SizeVector& sizes, ncvslideio::InputOutputArray a, WarmUpType wtype);

    class _declareHelper
    {
    public:
        _declareHelper& in(ncvslideio::InputOutputArray a1, WarmUpType wtype = WARMUP_READ);
        _declareHelper& in(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, WarmUpType wtype = WARMUP_READ);
        _declareHelper& in(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, ncvslideio::InputOutputArray a3, WarmUpType wtype = WARMUP_READ);
        _declareHelper& in(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, ncvslideio::InputOutputArray a3, ncvslideio::InputOutputArray a4, WarmUpType wtype = WARMUP_READ);

        _declareHelper& out(ncvslideio::InputOutputArray a1, WarmUpType wtype = WARMUP_WRITE);
        _declareHelper& out(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, WarmUpType wtype = WARMUP_WRITE);
        _declareHelper& out(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, ncvslideio::InputOutputArray a3, WarmUpType wtype = WARMUP_WRITE);
        _declareHelper& out(ncvslideio::InputOutputArray a1, ncvslideio::InputOutputArray a2, ncvslideio::InputOutputArray a3, ncvslideio::InputOutputArray a4, WarmUpType wtype = WARMUP_WRITE);

        _declareHelper& iterations(unsigned int n);
        _declareHelper& time(double timeLimitSecs);
        _declareHelper& tbb_threads(int n = -1);
        _declareHelper& runs(unsigned int runsNumber);

        _declareHelper& strategy(enum PERF_STRATEGY s);
    private:
        TestBase* test;
        _declareHelper(TestBase* t);
        _declareHelper(const _declareHelper&);
        _declareHelper& operator=(const _declareHelper&);
        friend class TestBase;
    };
    friend class _declareHelper;

    bool verified;

public:
    _declareHelper declare;

    void setVerified() { this->verified = true; }
};

template<typename T> class TestBaseWithParam: public TestBase, public ::testing::WithParamInterface<T> {};

typedef tuple<ncvslideio::Size, MatType> Size_MatType_t;
typedef TestBaseWithParam<Size_MatType_t> Size_MatType;

/*****************************************************************************************\
*                              Print functions for googletest                             *
\*****************************************************************************************/
void PrintTo(const MatType& t, std::ostream* os);

} //namespace perf

namespace ncvslideio
{

void PrintTo(const String& str, ::std::ostream* os);
void PrintTo(const Size& sz, ::std::ostream* os);

} //namespace ncvslideio


/*****************************************************************************************\
*                        Macro definitions for performance tests                          *
\*****************************************************************************************/

#define CV__PERF_TEST_BODY_IMPL(name) \
    { \
       CV__TEST_NAMESPACE_CHECK \
       CV__TRACE_APP_FUNCTION_NAME("PERF_TEST: " name); \
       try { \
       ::cvtest::testSetUp(); \
       RunPerfTestBody(); \
       } \
       catch (cvtest::details::SkipTestExceptionBase& e) \
       { \
          printf("[     SKIP ] %s\n", e.what()); \
       } \
       ::cvtest::testTearDown(); \
    }

#define PERF_PROXY_NAMESPACE_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_perf_namespace_proxy

// Defines a performance test.
//
// The first parameter is the name of the test case, and the second
// parameter is the name of the test within the test case.
//
// The user should put his test code between braces after using this
// macro.  Example:
//
//   PERF_TEST(FooTest, InitializesCorrectly) {
//     Foo foo;
//     EXPECT_TRUE(foo.StatusIsOK());
//   }
#define PERF_TEST(test_case_name, test_name)\
    TEST_(test_case_name, test_name, ::perf::TestBase, PerfTestBody, CV_OVERRIDE, CV__PERF_TEST_BODY_IMPL)

// Defines a performance test that uses a test fixture.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// A test fixture class must be declared earlier.  The user should put
// his test code between braces after using this macro.  Example:
//
//   class FooTest : public ::perf::TestBase {
//    protected:
//     virtual void SetUp() { TestBase::SetUp(); b_.AddElement(3); }
//
//     Foo a_;
//     Foo b_;
//   };
//
//   PERF_TEST_F(FooTest, InitializesCorrectly) {
//     EXPECT_TRUE(a_.StatusIsOK());
//   }
//
//   PERF_TEST_F(FooTest, ReturnsElementCountCorrectly) {
//     EXPECT_EQ(0, a_.size());
//     EXPECT_EQ(1, b_.size());
//   }
#define PERF_TEST_F(fixture, testname) \
    namespace PERF_PROXY_NAMESPACE_NAME_(fixture, testname) {\
     class TestBase {/*compile error for this class means that you are trying to use perf::TestBase as a fixture*/};\
     class fixture : public ::fixture {\
      public:\
       fixture() {}\
      protected:\
       virtual void PerfTestBody() CV_OVERRIDE;\
     };\
     TEST_F(fixture, testname){ CV__PERF_TEST_BODY_IMPL(#fixture "_" #testname); }\
    }\
    void PERF_PROXY_NAMESPACE_NAME_(fixture, testname)::fixture::PerfTestBody()

// Defines a parametrized performance test.
//
// @Note PERF_TEST_P() below violates behavior of original Google Tests - there is no tests instantiation in original TEST_P()
// This macro is intended for usage with separate INSTANTIATE_TEST_CASE_P macro
#define PERF_TEST_P_(test_case_name, test_name) CV__TEST_P(test_case_name, test_name, PerfTestBody, CV_OVERRIDE, CV__PERF_TEST_BODY_IMPL)

// Defines a parametrized performance test.
//
// @Note Original TEST_P() macro doesn't instantiate tests with parameters. To keep original usage use PERF_TEST_P_() macro
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// The user should put his test code between braces after using this
// macro.  Example:
//
//   typedef ::perf::TestBaseWithParam<ncvslideio::Size> FooTest;
//
//   PERF_TEST_P(FooTest, DoTestingRight, ::testing::Values(::perf::szVGA, ::perf::sz720p) {
//     ncvslideio::Mat b(GetParam(), CV_8U, ncvslideio::Scalar(10));
//     ncvslideio::Mat a(GetParam(), CV_8U, ncvslideio::Scalar(20));
//     ncvslideio::Mat c(GetParam(), CV_8U, ncvslideio::Scalar(0));
//
//     declare.in(a, b).out(c).time(0.5);
//
//     TEST_CYCLE() ncvslideio::add(a, b, c);
//
//     SANITY_CHECK(c);
//   }
#define PERF_TEST_P(fixture, name, params)  \
    class fixture##_##name : public fixture {\
     public:\
      fixture##_##name() {}\
     protected:\
      virtual void PerfTestBody() CV_OVERRIDE;\
    };\
    CV__TEST_P(fixture##_##name, name, PerfTestBodyDummy,, CV__PERF_TEST_BODY_IMPL){} \
    INSTANTIATE_TEST_CASE_P(/*none*/, fixture##_##name, params);\
    void fixture##_##name::PerfTestBody()

#ifndef __CV_TEST_EXEC_ARGS
#if defined(_MSC_VER) && (_MSC_VER <= 1400)
#define __CV_TEST_EXEC_ARGS(...)    \
    while (++argc >= (--argc,-1)) {__VA_ARGS__; break;} /*this ugly construction is needed for VS 2005*/
#else
#define __CV_TEST_EXEC_ARGS(...)    \
    __VA_ARGS__;
#endif
#endif


#define CV_PERF_TEST_MAIN_INTERNALS(modulename, impls, ...)	\
    CV_TRACE_FUNCTION(); \
    { CV_TRACE_REGION("INIT"); \
    ::perf::Regression::Init(#modulename); \
    ::perf::TestBase::Init(std::vector<std::string>(impls, impls + sizeof impls / sizeof *impls), \
                           argc, argv); \
    ::testing::InitGoogleTest(&argc, argv); \
    ::testing::UnitTest::GetInstance()->listeners().Append(new cvtest::SystemInfoCollector); \
    ::testing::Test::RecordProperty("cv_module_name", #modulename); \
    ::perf::TestBase::RecordRunParameters(); \
    __CV_TEST_EXEC_ARGS(__VA_ARGS__) \
    } \
    return RUN_ALL_TESTS();

// impls must be an array, not a pointer; "plain" should always be one of the implementations
#define CV_PERF_TEST_MAIN_WITH_IMPLS(modulename, impls, ...) \
int main(int argc, char **argv)\
{\
    CV_PERF_TEST_MAIN_INTERNALS(modulename, impls, __VA_ARGS__)\
}

#define CV_PERF_TEST_MAIN(modulename, ...) \
int main(int argc, char **argv)\
{\
    const char * plain_only[] = { "plain" };\
    CV_PERF_TEST_MAIN_INTERNALS(modulename, plain_only, __VA_ARGS__)\
}

//! deprecated
#define TEST_CYCLE_N(n) for(declare.iterations(n); next() && startTimer(); stopTimer())
//! deprecated
#define TEST_CYCLE() for(; next() && startTimer(); stopTimer())
//! deprecated
#define TEST_CYCLE_MULTIRUN(runsNum) for(declare.runs(runsNum); next() && startTimer(); stopTimer()) for(int r = 0; r < runsNum; ++r)

#define PERF_SAMPLE_BEGIN() \
    for(; next() && startTimer(); stopTimer()) \
    { \
        CV_TRACE_REGION("iteration");
#define PERF_SAMPLE_END() \
    }

namespace perf
{
namespace comparators
{

template<typename T>
struct RectLess_
{
  bool operator()(const ncvslideio::Rect_<T>& r1, const ncvslideio::Rect_<T>& r2) const
  {
    return r1.x < r2.x ||
            (r1.x == r2.x && r1.y < r2.y) ||
            (r1.x == r2.x && r1.y == r2.y && r1.width < r2.width) ||
            (r1.x == r2.x && r1.y == r2.y && r1.width == r2.width && r1.height < r2.height);
  }
};

typedef RectLess_<int> RectLess;

struct KeypointGreater
{
    bool operator()(const ncvslideio::KeyPoint& kp1, const ncvslideio::KeyPoint& kp2) const
    {
        if (kp1.response > kp2.response) return true;
        if (kp1.response < kp2.response) return false;
        if (kp1.size > kp2.size) return true;
        if (kp1.size < kp2.size) return false;
        if (kp1.octave > kp2.octave) return true;
        if (kp1.octave < kp2.octave) return false;
        if (kp1.pt.y < kp2.pt.y) return false;
        if (kp1.pt.y > kp2.pt.y) return true;
        return kp1.pt.x < kp2.pt.x;
    }
};

} //namespace comparators

void sort(std::vector<ncvslideio::KeyPoint>& pts, ncvslideio::InputOutputArray descriptors);
} //namespace perf

#endif //OPENCV_TS_PERF_HPP
