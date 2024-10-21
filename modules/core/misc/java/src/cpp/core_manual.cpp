#define LOG_TAG "org.opencv.core.Core"
#include "common.h"
#include "core_manual.hpp"
#include "opencv2/core/utility.hpp"

static int quietCallback( int, const char*, const char*, const char*, int, void* )
{
    return 0;
}

namespace cv {

void setErrorVerbosity(bool verbose)
{
    if(verbose)
        ncvslideio::redirectError(0);
    else
        ncvslideio::redirectError((ncvslideio::ErrorCallback)quietCallback);
}

}
