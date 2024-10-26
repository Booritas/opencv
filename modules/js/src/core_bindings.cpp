/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// Author: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015 The Regents of the University of California (Regents)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//M*/

#include <emscripten/bind.h>

@INCLUDES@
#include "../../../modules/core/src/parallel_impl.hpp"

#ifdef TEST_WASM_INTRIN
#include "../../../modules/core/include/opencv2/core/hal/intrin.hpp"
#include "../../../modules/core/include/opencv2/core/utils/trace.hpp"
#include "../../../modules/ts/include/opencv2/ts/ts_gtest.h"
namespace ncvslideio {
namespace hal {
#include "../../../modules/core/test/test_intrin_utils.hpp"
}
}
#endif

using namespace emscripten;
using namespace ncvslideio;

using namespace ncvslideio::segmentation;  // FIXIT

#ifdef HAVE_OPENCV_OBJDETECT
using namespace ncvslideio::aruco;
typedef aruco::DetectorParameters aruco_DetectorParameters;
typedef QRCodeDetectorAruco::Params QRCodeDetectorAruco_Params;
#endif

#ifdef HAVE_OPENCV_DNN
using namespace ncvslideio::dnn;
#endif

#ifdef HAVE_OPENCV_FEATURES2D
typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;
#endif

#ifdef HAVE_OPENCV_VIDEO
typedef TrackerMIL::Params TrackerMIL_Params;
#endif

// HACK: JS generator ommits namespace for parameter types for some reason. Added typedef to handle std::string correctly
typedef std::string string;

namespace binding_utils
{
    template<typename classT, typename enumT>
    static inline typename std::underlying_type<enumT>::type classT::* underlying_ptr(enumT classT::* enum_ptr)
    {
        return reinterpret_cast<typename std::underlying_type<enumT>::type classT::*>(enum_ptr);
    }

    template<typename T>
    emscripten::val matData(const ncvslideio::Mat& mat)
    {
        return emscripten::val(emscripten::memory_view<T>((mat.total()*mat.elemSize())/sizeof(T),
                               (T*)mat.data));
    }

    template<typename T>
    emscripten::val matPtr(const ncvslideio::Mat& mat, int i)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(0), mat.ptr<T>(i)));
    }

    template<typename T>
    emscripten::val matPtr(const ncvslideio::Mat& mat, int i, int j)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(1), mat.ptr<T>(i,j)));
    }

    ncvslideio::Mat* createMat(int rows, int cols, int type, intptr_t data, size_t step)
    {
        return new ncvslideio::Mat(rows, cols, type, reinterpret_cast<void*>(data), step);
    }

    static emscripten::val getMatSize(const ncvslideio::Mat& mat)
    {
        emscripten::val size = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            size.call<void>("push", mat.size[i]);
        }
        return size;
    }

    static emscripten::val getMatStep(const ncvslideio::Mat& mat)
    {
        emscripten::val step = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            step.call<void>("push", mat.step[i]);
        }
        return step;
    }

    static Mat matEye(int rows, int cols, int type)
    {
        return Mat(ncvslideio::Mat::eye(rows, cols, type));
    }

    static Mat matEye(Size size, int type)
    {
        return Mat(ncvslideio::Mat::eye(size, type));
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta)
    {
        obj.convertTo(m, rtype, alpha, beta);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype)
    {
        obj.convertTo(m, rtype);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha)
    {
        obj.convertTo(m, rtype, alpha);
    }

    Size matSize(const ncvslideio::Mat& mat)
    {
        return mat.size();
    }

    ncvslideio::Mat matZeros(int arg0, int arg1, int arg2)
    {
        return ncvslideio::Mat::zeros(arg0, arg1, arg2);
    }

    ncvslideio::Mat matZeros(ncvslideio::Size arg0, int arg1)
    {
        return ncvslideio::Mat::zeros(arg0,arg1);
    }

    ncvslideio::Mat matOnes(int arg0, int arg1, int arg2)
    {
        return ncvslideio::Mat::ones(arg0, arg1, arg2);
    }

    ncvslideio::Mat matOnes(ncvslideio::Size arg0, int arg1)
    {
        return ncvslideio::Mat::ones(arg0, arg1);
    }

    double matDot(const ncvslideio::Mat& obj, const Mat& mat)
    {
        return  obj.dot(mat);
    }

    Mat matMul(const ncvslideio::Mat& obj, const Mat& mat, double scale)
    {
        return  Mat(obj.mul(mat, scale));
    }

    Mat matT(const ncvslideio::Mat& obj)
    {
        return  Mat(obj.t());
    }

    Mat matInv(const ncvslideio::Mat& obj, int type)
    {
        return  Mat(obj.inv(type));
    }

    void matCopyTo(const ncvslideio::Mat& obj, ncvslideio::Mat& mat)
    {
        return obj.copyTo(mat);
    }

    void matCopyTo(const ncvslideio::Mat& obj, ncvslideio::Mat& mat, const ncvslideio::Mat& mask)
    {
        return obj.copyTo(mat, mask);
    }

    Mat matDiag(const ncvslideio::Mat& obj, int d)
    {
        return obj.diag(d);
    }

    Mat matDiag(const ncvslideio::Mat& obj)
    {
        return obj.diag();
    }

    void matSetTo(ncvslideio::Mat& obj, const ncvslideio::Scalar& s)
    {
        obj.setTo(s);
    }

    void matSetTo(ncvslideio::Mat& obj, const ncvslideio::Scalar& s, const ncvslideio::Mat& mask)
    {
        obj.setTo(s, mask);
    }

    emscripten::val rotatedRectPoints(const ncvslideio::RotatedRect& obj)
    {
        ncvslideio::Point2f points[4];
        obj.points(points);
        emscripten::val pointsArray = emscripten::val::array();
        for (int i = 0; i < 4; i++) {
            pointsArray.call<void>("push", points[i]);
        }
        return pointsArray;
    }

    Rect rotatedRectBoundingRect(const ncvslideio::RotatedRect& obj)
    {
        return obj.boundingRect();
    }

    Rect2f rotatedRectBoundingRect2f(const ncvslideio::RotatedRect& obj)
    {
        return obj.boundingRect2f();
    }

    int cvMatDepth(int flags)
    {
        return CV_MAT_DEPTH(flags);
    }

    class MinMaxLoc
    {
    public:
        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;
    };

    MinMaxLoc minMaxLoc(const ncvslideio::Mat& src, const ncvslideio::Mat& mask)
    {
        MinMaxLoc result;
        ncvslideio::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc, mask);
        return result;
    }

    MinMaxLoc minMaxLoc_1(const ncvslideio::Mat& src)
    {
        MinMaxLoc result;
        ncvslideio::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc);
        return result;
    }

    class Circle
    {
    public:
        Point2f center;
        float radius;
    };

#ifdef HAVE_OPENCV_IMGPROC
    Circle minEnclosingCircle(const ncvslideio::Mat& points)
    {
        Circle circle;
        ncvslideio::minEnclosingCircle(points, circle.center, circle.radius);
        return circle;
    }

    int floodFill_withRect_helper(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6 = Scalar(), Scalar arg7 = Scalar(), int arg8 = 4)
    {
        ncvslideio::Rect rect;

        int rc = ncvslideio::floodFill(arg1, arg2, arg3, arg4, &rect, arg6, arg7, arg8);

        arg5.set("x", emscripten::val(rect.x));
        arg5.set("y", emscripten::val(rect.y));
        arg5.set("width", emscripten::val(rect.width));
        arg5.set("height", emscripten::val(rect.height));

        return rc;
    }

    int floodFill_wrapper(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6, Scalar arg7, int arg8) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    int floodFill_wrapper_1(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6, Scalar arg7) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    int floodFill_wrapper_2(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    int floodFill_wrapper_3(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5);
    }

    int floodFill_wrapper_4(ncvslideio::Mat& arg1, ncvslideio::Mat& arg2, Point arg3, Scalar arg4) {
        return ncvslideio::floodFill(arg1, arg2, arg3, arg4);
    }
#endif

#ifdef HAVE_OPENCV_VIDEO
    emscripten::val CamShiftWrapper(const ncvslideio::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        RotatedRect rotatedRect = ncvslideio::CamShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", rotatedRect);
        result.call<void>("push", arg2);
        return result;
    }

    emscripten::val meanShiftWrapper(const ncvslideio::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        int n = ncvslideio::meanShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", n);
        result.call<void>("push", arg2);
        return result;
    }


    void Tracker_init_wrapper(ncvslideio::Tracker& arg0, const ncvslideio::Mat& arg1, const Rect& arg2)
    {
        return arg0.init(arg1, arg2);
    }

    emscripten::val Tracker_update_wrapper(ncvslideio::Tracker& arg0, const ncvslideio::Mat& arg1)
    {
        Rect rect;
        bool update = arg0.update(arg1, rect);

        emscripten::val result = emscripten::val::array();
        result.call<void>("push", update);
        result.call<void>("push", rect);
        return result;
    }
#endif  // HAVE_OPENCV_VIDEO

    std::string getExceptionMsg(const ncvslideio::Exception& e) {
        return e.msg;
    }

    void setExceptionMsg(ncvslideio::Exception& e, std::string msg) {
        e.msg = msg;
        return;
    }

    ncvslideio::Exception exceptionFromPtr(intptr_t ptr) {
        return *reinterpret_cast<ncvslideio::Exception*>(ptr);
    }

    std::string getBuildInformation() {
        return ncvslideio::getBuildInformation();
    }

#ifdef TEST_WASM_INTRIN
    void test_hal_intrin_uint8() {
        ncvslideio::hal::test_hal_intrin_uint8();
    }
    void test_hal_intrin_int8() {
        ncvslideio::hal::test_hal_intrin_int8();
    }
    void test_hal_intrin_uint16() {
        ncvslideio::hal::test_hal_intrin_uint16();
    }
    void test_hal_intrin_int16() {
        ncvslideio::hal::test_hal_intrin_int16();
    }
    void test_hal_intrin_uint32() {
        ncvslideio::hal::test_hal_intrin_uint32();
    }
    void test_hal_intrin_int32() {
        ncvslideio::hal::test_hal_intrin_int32();
    }
    void test_hal_intrin_uint64() {
        ncvslideio::hal::test_hal_intrin_uint64();
    }
    void test_hal_intrin_int64() {
        ncvslideio::hal::test_hal_intrin_int64();
    }
    void test_hal_intrin_float32() {
        ncvslideio::hal::test_hal_intrin_float32();
    }
    void test_hal_intrin_float64() {
        ncvslideio::hal::test_hal_intrin_float64();
    }
    void test_hal_intrin_all() {
        ncvslideio::hal::test_hal_intrin_uint8();
        ncvslideio::hal::test_hal_intrin_int8();
        ncvslideio::hal::test_hal_intrin_uint16();
        ncvslideio::hal::test_hal_intrin_int16();
        ncvslideio::hal::test_hal_intrin_uint32();
        ncvslideio::hal::test_hal_intrin_int32();
        ncvslideio::hal::test_hal_intrin_uint64();
        ncvslideio::hal::test_hal_intrin_int64();
        ncvslideio::hal::test_hal_intrin_float32();
        ncvslideio::hal::test_hal_intrin_float64();
    }
#endif
}

EMSCRIPTEN_BINDINGS(binding_utils)
{
    register_vector<int>("IntVector");
    register_vector<char>("CharVector");
    register_vector<float>("FloatVector");
    register_vector<double>("DoubleVector");
    register_vector<std::string>("StringVector");
    register_vector<ncvslideio::Point>("PointVector");
    register_vector<ncvslideio::Mat>("MatVector");
    register_vector<ncvslideio::Rect>("RectVector");
    register_vector<ncvslideio::KeyPoint>("KeyPointVector");
    register_vector<ncvslideio::DMatch>("DMatchVector");
    register_vector<std::vector<ncvslideio::DMatch>>("DMatchVectorVector");


    emscripten::class_<ncvslideio::Mat>("Mat")
        .constructor<>()
        .constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor<int, int, int, const Scalar&>()
        .constructor(&binding_utils::createMat, allow_raw_pointers())

        .class_function("eye", select_overload<Mat(Size, int)>(&binding_utils::matEye))
        .class_function("eye", select_overload<Mat(int, int, int)>(&binding_utils::matEye))
        .class_function("ones", select_overload<Mat(Size, int)>(&binding_utils::matOnes))
        .class_function("ones", select_overload<Mat(int, int, int)>(&binding_utils::matOnes))
        .class_function("zeros", select_overload<Mat(Size, int)>(&binding_utils::matZeros))
        .class_function("zeros", select_overload<Mat(int, int, int)>(&binding_utils::matZeros))

        .property("rows", &ncvslideio::Mat::rows)
        .property("cols", &ncvslideio::Mat::cols)
        .property("matSize", &binding_utils::getMatSize)
        .property("step", &binding_utils::getMatStep)
        .property("data", &binding_utils::matData<unsigned char>)
        .property("data8S", &binding_utils::matData<char>)
        .property("data16U", &binding_utils::matData<unsigned short>)
        .property("data16S", &binding_utils::matData<short>)
        .property("data32S", &binding_utils::matData<int>)
        .property("data32F", &binding_utils::matData<float>)
        .property("data64F", &binding_utils::matData<double>)

        .function("elemSize", select_overload<size_t()const>(&ncvslideio::Mat::elemSize))
        .function("elemSize1", select_overload<size_t()const>(&ncvslideio::Mat::elemSize1))
        .function("channels", select_overload<int()const>(&ncvslideio::Mat::channels))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double, double)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double)>(&binding_utils::convertTo))
        .function("total", select_overload<size_t()const>(&ncvslideio::Mat::total))
        .function("row", select_overload<Mat(int)const>(&ncvslideio::Mat::row))
        .function("create", select_overload<void(int, int, int)>(&ncvslideio::Mat::create))
        .function("create", select_overload<void(Size, int)>(&ncvslideio::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&ncvslideio::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&ncvslideio::Mat::rowRange))
        .function("copyTo", select_overload<void(const Mat&, Mat&)>(&binding_utils::matCopyTo))
        .function("copyTo", select_overload<void(const Mat&, Mat&, const Mat&)>(&binding_utils::matCopyTo))
        .function("type", select_overload<int()const>(&ncvslideio::Mat::type))
        .function("empty", select_overload<bool()const>(&ncvslideio::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&ncvslideio::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&ncvslideio::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&ncvslideio::Mat::step1))
        .function("clone", select_overload<Mat()const>(&ncvslideio::Mat::clone))
        .function("depth", select_overload<int()const>(&ncvslideio::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&ncvslideio::Mat::col))
        .function("dot", select_overload<double(const Mat&, const Mat&)>(&binding_utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&binding_utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&binding_utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&binding_utils::matT))
        .function("roi", select_overload<Mat(const Rect&)const>(&ncvslideio::Mat::operator()))
        .function("diag", select_overload<Mat(const Mat&, int)>(&binding_utils::matDiag))
        .function("diag", select_overload<Mat(const Mat&)>(&binding_utils::matDiag))
        .function("isContinuous", select_overload<bool()const>(&ncvslideio::Mat::isContinuous))
        .function("setTo", select_overload<void(Mat&, const Scalar&)>(&binding_utils::matSetTo))
        .function("setTo", select_overload<void(Mat&, const Scalar&, const Mat&)>(&binding_utils::matSetTo))
        .function("size", select_overload<Size(const Mat&)>(&binding_utils::matSize))

        .function("ptr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("charPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<char>))
        .function("charPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<char>))
        .function("shortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<short>))
        .function("shortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<short>))
        .function("ushortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned short>))
        .function("ushortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned short>))
        .function("intPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<int>))
        .function("intPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<int>))
        .function("floatPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<float>))
        .function("floatPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<float>))
        .function("doublePtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<double>))
        .function("doublePtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<double>))

        .function("charAt", select_overload<char&(int)>(&ncvslideio::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int)>(&ncvslideio::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int, int)>(&ncvslideio::Mat::at<char>))
        .function("ucharAt", select_overload<unsigned char&(int)>(&ncvslideio::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int)>(&ncvslideio::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int, int)>(&ncvslideio::Mat::at<unsigned char>))
        .function("shortAt", select_overload<short&(int)>(&ncvslideio::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int)>(&ncvslideio::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int, int)>(&ncvslideio::Mat::at<short>))
        .function("ushortAt", select_overload<unsigned short&(int)>(&ncvslideio::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int)>(&ncvslideio::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int, int)>(&ncvslideio::Mat::at<unsigned short>))
        .function("intAt", select_overload<int&(int)>(&ncvslideio::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int)>(&ncvslideio::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int, int)>(&ncvslideio::Mat::at<int>) )
        .function("floatAt", select_overload<float&(int)>(&ncvslideio::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int)>(&ncvslideio::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int, int)>(&ncvslideio::Mat::at<float>))
        .function("doubleAt", select_overload<double&(int, int, int)>(&ncvslideio::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int)>(&ncvslideio::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int, int)>(&ncvslideio::Mat::at<double>));

    emscripten::value_object<ncvslideio::Range>("Range")
        .field("start", &ncvslideio::Range::start)
        .field("end", &ncvslideio::Range::end);

    emscripten::value_object<ncvslideio::TermCriteria>("TermCriteria")
        .field("type", &ncvslideio::TermCriteria::type)
        .field("maxCount", &ncvslideio::TermCriteria::maxCount)
        .field("epsilon", &ncvslideio::TermCriteria::epsilon);

#define EMSCRIPTEN_CV_SIZE(type) \
    emscripten::value_object<type>("#type") \
        .field("width", &type::width) \
        .field("height", &type::height);

    EMSCRIPTEN_CV_SIZE(Size)
    EMSCRIPTEN_CV_SIZE(Size2f)

#define EMSCRIPTEN_CV_POINT(type) \
    emscripten::value_object<type>("#type") \
        .field("x", &type::x) \
        .field("y", &type::y); \

    EMSCRIPTEN_CV_POINT(Point)
    EMSCRIPTEN_CV_POINT(Point2f)

#define EMSCRIPTEN_CV_RECT(type, name) \
    emscripten::value_object<ncvslideio::Rect_<type>> (name) \
        .field("x", &ncvslideio::Rect_<type>::x) \
        .field("y", &ncvslideio::Rect_<type>::y) \
        .field("width", &ncvslideio::Rect_<type>::width) \
        .field("height", &ncvslideio::Rect_<type>::height);

    EMSCRIPTEN_CV_RECT(int, "Rect")
    EMSCRIPTEN_CV_RECT(float, "Rect2f")

    emscripten::value_object<ncvslideio::RotatedRect>("RotatedRect")
        .field("center", &ncvslideio::RotatedRect::center)
        .field("size", &ncvslideio::RotatedRect::size)
        .field("angle", &ncvslideio::RotatedRect::angle);

    function("rotatedRectPoints", select_overload<emscripten::val(const ncvslideio::RotatedRect&)>(&binding_utils::rotatedRectPoints));
    function("rotatedRectBoundingRect", select_overload<Rect(const ncvslideio::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect));
    function("rotatedRectBoundingRect2f", select_overload<Rect2f(const ncvslideio::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect2f));

    emscripten::value_object<ncvslideio::KeyPoint>("KeyPoint")
        .field("angle", &ncvslideio::KeyPoint::angle)
        .field("class_id", &ncvslideio::KeyPoint::class_id)
        .field("octave", &ncvslideio::KeyPoint::octave)
        .field("pt", &ncvslideio::KeyPoint::pt)
        .field("response", &ncvslideio::KeyPoint::response)
        .field("size", &ncvslideio::KeyPoint::size);

    emscripten::value_object<ncvslideio::DMatch>("DMatch")
        .field("queryIdx", &ncvslideio::DMatch::queryIdx)
        .field("trainIdx", &ncvslideio::DMatch::trainIdx)
        .field("imgIdx", &ncvslideio::DMatch::imgIdx)
        .field("distance", &ncvslideio::DMatch::distance);

    emscripten::value_array<ncvslideio::Scalar_<double>> ("Scalar")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>())
        .element(emscripten::index<3>());

    emscripten::value_object<binding_utils::MinMaxLoc>("MinMaxLoc")
        .field("minVal", &binding_utils::MinMaxLoc::minVal)
        .field("maxVal", &binding_utils::MinMaxLoc::maxVal)
        .field("minLoc", &binding_utils::MinMaxLoc::minLoc)
        .field("maxLoc", &binding_utils::MinMaxLoc::maxLoc);

    emscripten::value_object<binding_utils::Circle>("Circle")
        .field("center", &binding_utils::Circle::center)
        .field("radius", &binding_utils::Circle::radius);

    emscripten::value_object<ncvslideio::Moments >("Moments")
        .field("m00", &ncvslideio::Moments::m00)
        .field("m10", &ncvslideio::Moments::m10)
        .field("m01", &ncvslideio::Moments::m01)
        .field("m20", &ncvslideio::Moments::m20)
        .field("m11", &ncvslideio::Moments::m11)
        .field("m02", &ncvslideio::Moments::m02)
        .field("m30", &ncvslideio::Moments::m30)
        .field("m21", &ncvslideio::Moments::m21)
        .field("m12", &ncvslideio::Moments::m12)
        .field("m03", &ncvslideio::Moments::m03)
        .field("mu20", &ncvslideio::Moments::mu20)
        .field("mu11", &ncvslideio::Moments::mu11)
        .field("mu02", &ncvslideio::Moments::mu02)
        .field("mu30", &ncvslideio::Moments::mu30)
        .field("mu21", &ncvslideio::Moments::mu21)
        .field("mu12", &ncvslideio::Moments::mu12)
        .field("mu03", &ncvslideio::Moments::mu03)
        .field("nu20", &ncvslideio::Moments::nu20)
        .field("nu11", &ncvslideio::Moments::nu11)
        .field("nu02", &ncvslideio::Moments::nu02)
        .field("nu30", &ncvslideio::Moments::nu30)
        .field("nu21", &ncvslideio::Moments::nu21)
        .field("nu12", &ncvslideio::Moments::nu12)
        .field("nu03", &ncvslideio::Moments::nu03);

    emscripten::value_object<ncvslideio::Exception>("Exception")
        .field("code", &ncvslideio::Exception::code)
        .field("msg", &binding_utils::getExceptionMsg, &binding_utils::setExceptionMsg);

    function("exceptionFromPtr", &binding_utils::exceptionFromPtr, allow_raw_pointers());

#ifdef HAVE_OPENCV_IMGPROC
    function("minEnclosingCircle", select_overload<binding_utils::Circle(const ncvslideio::Mat&)>(&binding_utils::minEnclosingCircle));

    function("floodFill", select_overload<int(ncvslideio::Mat&, ncvslideio::Mat&, Point, Scalar, emscripten::val, Scalar, Scalar, int)>(&binding_utils::floodFill_wrapper));

    function("floodFill", select_overload<int(ncvslideio::Mat&, ncvslideio::Mat&, Point, Scalar, emscripten::val, Scalar, Scalar)>(&binding_utils::floodFill_wrapper_1));

    function("floodFill", select_overload<int(ncvslideio::Mat&, ncvslideio::Mat&, Point, Scalar, emscripten::val, Scalar)>(&binding_utils::floodFill_wrapper_2));

    function("floodFill", select_overload<int(ncvslideio::Mat&, ncvslideio::Mat&, Point, Scalar, emscripten::val)>(&binding_utils::floodFill_wrapper_3));

    function("floodFill", select_overload<int(ncvslideio::Mat&, ncvslideio::Mat&, Point, Scalar)>(&binding_utils::floodFill_wrapper_4));
#endif

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const ncvslideio::Mat&, const ncvslideio::Mat&)>(&binding_utils::minMaxLoc));

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const ncvslideio::Mat&)>(&binding_utils::minMaxLoc_1));

#ifdef HAVE_OPENCV_IMGPROC
    function("morphologyDefaultBorderValue", &ncvslideio::morphologyDefaultBorderValue);
#endif

    function("CV_MAT_DEPTH", &binding_utils::cvMatDepth);

#ifdef HAVE_OPENCV_VIDEO
    function("CamShift", select_overload<emscripten::val(const ncvslideio::Mat&, Rect&, TermCriteria)>(&binding_utils::CamShiftWrapper));

    function("meanShift", select_overload<emscripten::val(const ncvslideio::Mat&, Rect&, TermCriteria)>(&binding_utils::meanShiftWrapper));

    emscripten::class_<ncvslideio::Tracker >("Tracker")
        .function("init", select_overload<void(ncvslideio::Tracker&,const ncvslideio::Mat&,const Rect&)>(&binding_utils::Tracker_init_wrapper), pure_virtual())
        .function("update", select_overload<emscripten::val(ncvslideio::Tracker&,const ncvslideio::Mat&)>(&binding_utils::Tracker_update_wrapper), pure_virtual());

#endif

    function("getBuildInformation", &binding_utils::getBuildInformation);

#ifdef HAVE_PTHREADS_PF
    function("parallel_pthreads_set_threads_num", &ncvslideio::parallel_pthreads_set_threads_num);
    function("parallel_pthreads_get_threads_num", &ncvslideio::parallel_pthreads_get_threads_num);
#endif

#ifdef TEST_WASM_INTRIN
    function("test_hal_intrin_uint8", &binding_utils::test_hal_intrin_uint8);
    function("test_hal_intrin_int8", &binding_utils::test_hal_intrin_int8);
    function("test_hal_intrin_uint16", &binding_utils::test_hal_intrin_uint16);
    function("test_hal_intrin_int16", &binding_utils::test_hal_intrin_int16);
    function("test_hal_intrin_uint32", &binding_utils::test_hal_intrin_uint32);
    function("test_hal_intrin_int32", &binding_utils::test_hal_intrin_int32);
    function("test_hal_intrin_uint64", &binding_utils::test_hal_intrin_uint64);
    function("test_hal_intrin_int64", &binding_utils::test_hal_intrin_int64);
    function("test_hal_intrin_float32", &binding_utils::test_hal_intrin_float32);
    function("test_hal_intrin_float64", &binding_utils::test_hal_intrin_float64);
    function("test_hal_intrin_all", &binding_utils::test_hal_intrin_all);
#endif

    constant("CV_8UC1", CV_8UC1);
    constant("CV_8UC2", CV_8UC2);
    constant("CV_8UC3", CV_8UC3);
    constant("CV_8UC4", CV_8UC4);

    constant("CV_8SC1", CV_8SC1);
    constant("CV_8SC2", CV_8SC2);
    constant("CV_8SC3", CV_8SC3);
    constant("CV_8SC4", CV_8SC4);

    constant("CV_16UC1", CV_16UC1);
    constant("CV_16UC2", CV_16UC2);
    constant("CV_16UC3", CV_16UC3);
    constant("CV_16UC4", CV_16UC4);

    constant("CV_16SC1", CV_16SC1);
    constant("CV_16SC2", CV_16SC2);
    constant("CV_16SC3", CV_16SC3);
    constant("CV_16SC4", CV_16SC4);

    constant("CV_32SC1", CV_32SC1);
    constant("CV_32SC2", CV_32SC2);
    constant("CV_32SC3", CV_32SC3);
    constant("CV_32SC4", CV_32SC4);

    constant("CV_32FC1", CV_32FC1);
    constant("CV_32FC2", CV_32FC2);
    constant("CV_32FC3", CV_32FC3);
    constant("CV_32FC4", CV_32FC4);

    constant("CV_64FC1", CV_64FC1);
    constant("CV_64FC2", CV_64FC2);
    constant("CV_64FC3", CV_64FC3);
    constant("CV_64FC4", CV_64FC4);

    constant("CV_8U", CV_8U);
    constant("CV_8S", CV_8S);
    constant("CV_16U", CV_16U);
    constant("CV_16S", CV_16S);
    constant("CV_32S",  CV_32S);
    constant("CV_32F", CV_32F);
    constant("CV_64F", CV_64F);

    constant("INT_MIN", INT_MIN);
    constant("INT_MAX", INT_MAX);
}
