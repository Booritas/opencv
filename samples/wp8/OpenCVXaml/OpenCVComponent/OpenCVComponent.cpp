// OpenCVComponent.cpp
#include "pch.h"
#include "OpenCVComponent.h"

#include <opencv2\imgproc\types_c.h>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>
#include <algorithm>

using namespace OpenCVComponent;
using namespace Platform;
using namespace concurrency;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;

void CopyIVectorToMatrix(IVector<int>^ input, ncvslideio::Mat& mat, int size);
void CopyMatrixToVector(const ncvslideio::Mat& mat, std::vector<int>& vector, int size);

OpenCVLib::OpenCVLib()
{
}

IAsyncOperation<IVectorView<int>^>^ OpenCVLib::ProcessAsync(IVector<int>^ input, int width, int height)
{
    int size = input->Size;
    ncvslideio::Mat mat(width, height, CV_8UC4);
    CopyIVectorToMatrix(input, mat, size);

    return create_async([=]() -> IVectorView<int>^
    {
        // convert to grayscale
        ncvslideio::Mat intermediateMat;
        ncvslideio::cvtColor(mat, intermediateMat, COLOR_RGB2GRAY);

        // convert to BGRA
        ncvslideio::cvtColor(intermediateMat, mat, COLOR_GRAY2BGRA);

        std::vector<int> output;
        CopyMatrixToVector(mat, output, size);

        // Return the outputs as a VectorView<float>
        return ref new Platform::Collections::VectorView<int>(output);
    });
}


void CopyIVectorToMatrix(IVector<int>^ input, ncvslideio::Mat& mat, int size)
{
    unsigned char* data = mat.data;
    for (int i = 0; i < size; i++)
    {
        int value = input->GetAt(i);
        memcpy(data, (void*) &value, 4);
        data += 4;
    }
}

void CopyMatrixToVector(const ncvslideio::Mat& mat, std::vector<int>& vector, int size)
{
    int* data = (int*) mat.data;
    for (int i = 0; i < size; i++)
    {
        vector.push_back(data[i]);
    }
}
