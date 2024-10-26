// [filter2d_api]
#include <opencv2/gapi.hpp>

G_TYPED_KERNEL(GFilter2D,
               <ncvslideio::GMat(ncvslideio::GMat,int,ncvslideio::Mat,ncvslideio::Point,double,int,ncvslideio::Scalar)>,
               "org.opencv.imgproc.filters.filter2D")
{
    static ncvslideio::GMatDesc                 // outMeta's return value type
    outMeta(ncvslideio::GMatDesc    in       ,  // descriptor of input GMat
            int             ddepth   ,  // depth parameter
            ncvslideio::Mat      /* coeffs */,  // (unused)
            ncvslideio::Point    /* anchor */,  // (unused)
            double       /* scale  */,  // (unused)
            int          /* border */,  // (unused)
            ncvslideio::Scalar   /* bvalue */ ) // (unused)
    {
        return in.withDepth(ddepth);
    }
};
// [filter2d_api]

ncvslideio::GMat filter2D(ncvslideio::GMat  ,
                  int       ,
                  ncvslideio::Mat   ,
                  ncvslideio::Point ,
                  double    ,
                  int       ,
                  ncvslideio::Scalar);

// [filter2d_wrap]
ncvslideio::GMat filter2D(ncvslideio::GMat   in,
                  int        ddepth,
                  ncvslideio::Mat    k,
                  ncvslideio::Point  anchor  = ncvslideio::Point(-1,-1),
                  double     scale   = 0.,
                  int        border  = ncvslideio::BORDER_DEFAULT,
                  ncvslideio::Scalar bval    = ncvslideio::Scalar(0))
{
    return GFilter2D::on(in, ddepth, k, anchor, scale, border, bval);
}
// [filter2d_wrap]

// [compound]
#include <opencv2/gapi/gcompoundkernel.hpp>       // GAPI_COMPOUND_KERNEL()

using PointArray2f = ncvslideio::GArray<ncvslideio::Point2f>;

G_TYPED_KERNEL(HarrisCorners,
               <PointArray2f(ncvslideio::GMat,int,double,double,int,double)>,
               "org.opencv.imgproc.harris_corner")
{
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc &,
                                  int,
                                  double,
                                  double,
                                  int,
                                  double)
    {
        // No special metadata for arrays in G-API (yet)
        return ncvslideio::empty_array_desc();
    }
};

// Define Fluid-backend-local kernels which form GoodFeatures
G_TYPED_KERNEL(HarrisResponse,
               <ncvslideio::GMat(ncvslideio::GMat,double,int,double)>,
               "org.opencv.fluid.harris_response")
{
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in,
                                double,
                                int,
                                double)
    {
        return in.withType(CV_32F, 1);
    }
};

G_TYPED_KERNEL(ArrayNMS,
               <PointArray2f(ncvslideio::GMat,int,double)>,
               "org.opencv.cpu.nms_array")
{
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc &,
                                  int,
                                  double)
    {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_COMPOUND_KERNEL(GFluidHarrisCorners, HarrisCorners)
{
    static PointArray2f
    expand(ncvslideio::GMat in,
           int      maxCorners,
           double   quality,
           double   minDist,
           int      blockSize,
           double   k)
    {
        ncvslideio::GMat response = HarrisResponse::on(in, quality, blockSize, k);
        return ArrayNMS::on(response, maxCorners, minDist);
    }
};

// Then implement HarrisResponse as Fluid kernel and NMSresponse
// as a generic (OpenCV) kernel
// [compound]

// [filter2d_ocv]
#include <opencv2/gapi/cpu/gcpukernel.hpp>     // GAPI_OCV_KERNEL()
#include <opencv2/imgproc.hpp>                 // ncvslideio::filter2D()

GAPI_OCV_KERNEL(GCPUFilter2D, GFilter2D)
{
    static void
    run(const ncvslideio::Mat    &in,       // in - derived from GMat
        const int         ddepth,   // opaque (passed as-is)
        const ncvslideio::Mat    &k,        // opaque (passed as-is)
        const ncvslideio::Point  &anchor,   // opaque (passed as-is)
        const double      delta,    // opaque (passed as-is)
        const int         border,   // opaque (passed as-is)
        const ncvslideio::Scalar &,         // opaque (passed as-is)
        ncvslideio::Mat          &out)      // out - derived from GMat (retval)
    {
        ncvslideio::filter2D(in, out, ddepth, k, anchor, delta, border);
    }
};
// [filter2d_ocv]

int main(int, char *[])
{
    std::cout << "This sample is non-complete. It is used as code snippents in documentation." << std::endl;

ncvslideio::Mat conv_kernel_mat;

{
// [filter2d_on]
ncvslideio::GMat in;
ncvslideio::GMat out = GFilter2D::on(/* GMat    */  in,
                             /* int     */  -1,
                             /* Mat     */  conv_kernel_mat,
                             /* Point   */  ncvslideio::Point(-1,-1),
                             /* double  */  0.,
                             /* int     */  ncvslideio::BORDER_DEFAULT,
                             /* Scalar  */  ncvslideio::Scalar(0));
// [filter2d_on]
}

{
// [filter2d_wrap_call]
ncvslideio::GMat in;
ncvslideio::GMat out = filter2D(in, -1, conv_kernel_mat);
// [filter2d_wrap_call]
}

return 0;
}
