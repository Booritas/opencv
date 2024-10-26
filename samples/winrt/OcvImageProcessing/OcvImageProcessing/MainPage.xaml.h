//
// MainPage.xaml.h
// Declaration of the MainPage class.
//

#pragma once

#include "MainPage.g.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>

namespace OcvImageProcessing
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public ref class MainPage sealed
    {
    public:
        MainPage();

    protected:
        virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;

    private:
        static const int PREVIEW  = 0;
        static const int GRAY     = 1;
        static const int CANNY    = 2;
        static const int BLUR     = 3;
        static const int FEATURES = 4;
        static const int SEPIA    = 5;

        void Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        ncvslideio::Mat ApplyGrayFilter(const ncvslideio::Mat& image);
        ncvslideio::Mat ApplyCannyFilter(const ncvslideio::Mat& image);
        ncvslideio::Mat ApplyBlurFilter(const ncvslideio::Mat& image);
        ncvslideio::Mat ApplyFindFeaturesFilter(const ncvslideio::Mat& image);
        ncvslideio::Mat ApplySepiaFilter(const ncvslideio::Mat& image);

        void UpdateImage(const ncvslideio::Mat& image);
        std::string CreateTempFile(const std::string &suffix);
        bool SaveImage(ncvslideio::Mat image);

        std::string StrToWStr(const std::wstring &wstr);
        ncvslideio::String ConvertPath(Platform::String^ path);

        ncvslideio::Mat Lena;
        unsigned int frameWidth, frameHeight;
    };
}
