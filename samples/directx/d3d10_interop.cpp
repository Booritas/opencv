/*
// A sample program demonstrating interoperability of OpenCV ncvslideio::UMat with Direct X surface
// At first, the data obtained from video file or camera and placed onto Direct X surface,
// following mapping of this Direct X surface to OpenCV ncvslideio::UMat and call ncvslideio::Blur function.
// The result is mapped back to Direct X surface and rendered through Direct X API.
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d10.h>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "d3dsample.hpp"


class D3D10WinApp : public D3DSample
{
public:
    D3D10WinApp(int width, int height, std::string& window_name, ncvslideio::VideoCapture& cap) :
        D3DSample(width, height, window_name, cap) {}

    ~D3D10WinApp() {}


    int create(void)
    {
        // base initialization
        D3DSample::create();

        // initialize DirectX
        HRESULT r;

        DXGI_SWAP_CHAIN_DESC scd;

        ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));

        scd.BufferCount       = 1;                               // one back buffer
        scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;      // use 32-bit color
        scd.BufferDesc.Width  = m_width;                         // set the back buffer width
        scd.BufferDesc.Height = m_height;                        // set the back buffer height
        scd.BufferUsage       = DXGI_USAGE_RENDER_TARGET_OUTPUT; // how swap chain is to be used
        scd.OutputWindow      = m_hWnd;                          // the window to be used
        scd.SampleDesc.Count  = 1;                               // how many multisamples
        scd.Windowed          = TRUE;                            // windowed/full-screen mode
        scd.SwapEffect        = DXGI_SWAP_EFFECT_DISCARD;
        scd.Flags             = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH; // allow full-screen switching

        r = ::D3D10CreateDeviceAndSwapChain(
                NULL,
                D3D10_DRIVER_TYPE_HARDWARE,
                NULL,
                0,
                D3D10_SDK_VERSION,
                &scd,
                &m_pD3D10SwapChain,
                &m_pD3D10Dev);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        r = m_pD3D10SwapChain->GetBuffer(0, __uuidof(ID3D10Texture2D), (LPVOID*)&m_pBackBuffer);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        r = m_pD3D10Dev->CreateRenderTargetView(m_pBackBuffer, NULL, &m_pRenderTarget);
        if (FAILED(r))
        {
            return EXIT_FAILURE;
        }

        m_pD3D10Dev->OMSetRenderTargets(1, &m_pRenderTarget, NULL);

        D3D10_VIEWPORT viewport;
        ZeroMemory(&viewport, sizeof(D3D10_VIEWPORT));

        viewport.Width    = m_width;
        viewport.Height   = m_height;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 0.0f;

        m_pD3D10Dev->RSSetViewports(1, &viewport);

        D3D10_TEXTURE2D_DESC desc = { 0 };

        desc.Width            = m_width;
        desc.Height           = m_height;
        desc.MipLevels        = 1;
        desc.ArraySize        = 1;
        desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags        = D3D10_BIND_SHADER_RESOURCE;
        desc.Usage            = D3D10_USAGE_DYNAMIC;
        desc.CPUAccessFlags   = D3D10_CPU_ACCESS_WRITE;

        r = m_pD3D10Dev->CreateTexture2D(&desc, NULL, &m_pSurface);
        if (FAILED(r))
        {
            std::cerr << "Can't create texture with input image" << std::endl;
            return EXIT_FAILURE;
        }

        // initialize OpenCL context of OpenCV lib from DirectX
        if (ncvslideio::ocl::haveOpenCL())
        {
            m_oclCtx = ncvslideio::directx::ocl::initializeContextFromD3D10Device(m_pD3D10Dev);
        }

        m_oclDevName = ncvslideio::ocl::useOpenCL() ?
            ncvslideio::ocl::Context::getDefault().device(0).name() :
            "No OpenCL device";

        return EXIT_SUCCESS;
    } // create()


    // get media data on DX surface for further processing
    int get_surface(ID3D10Texture2D** ppSurface)
    {
        HRESULT r;

        if (!m_cap.read(m_frame_bgr))
            return EXIT_FAILURE;

        ncvslideio::cvtColor(m_frame_bgr, m_frame_rgba, ncvslideio::COLOR_BGR2RGBA);

        UINT subResource = ::D3D10CalcSubresource(0, 0, 1);

        D3D10_MAPPED_TEXTURE2D mappedTex;
        r = m_pSurface->Map(subResource, D3D10_MAP_WRITE_DISCARD, 0, &mappedTex);
        if (FAILED(r))
        {
            return r;
        }

        ncvslideio::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);
        // copy video frame data to surface
        m_frame_rgba.copyTo(m);

        m_pSurface->Unmap(subResource);

        *ppSurface = m_pSurface;

        return EXIT_SUCCESS;
    } // get_surface()


    // process and render media data
    int render()
    {
        try
        {
            if (m_shutdown)
                return EXIT_SUCCESS;

            // capture user input once
            MODE mode = (m_mode == MODE_GPU_NV12) ? MODE_GPU_RGBA : m_mode;

            HRESULT r;
            ID3D10Texture2D* pSurface;

            r = get_surface(&pSurface);
            if (FAILED(r))
            {
                return EXIT_FAILURE;
            }

            m_timer.reset();
            m_timer.start();

            switch (mode)
            {
                case MODE_CPU:
                {
                    // process video frame on CPU
                    UINT subResource = ::D3D10CalcSubresource(0, 0, 1);

                    D3D10_MAPPED_TEXTURE2D mappedTex;
                    r = pSurface->Map(subResource, D3D10_MAP_WRITE_DISCARD, 0, &mappedTex);
                    if (FAILED(r))
                    {
                        return r;
                    }

                    ncvslideio::Mat m(m_height, m_width, CV_8UC4, mappedTex.pData, (int)mappedTex.RowPitch);

                    if (m_demo_processing)
                    {
                        // blur D3D10 surface with OpenCV on CPU
                        ncvslideio::blur(m, m, ncvslideio::Size(15, 15));
                    }

                    m_timer.stop();

                    ncvslideio::String strMode = ncvslideio::format("mode: %s", m_modeStr[MODE_CPU].c_str());
                    ncvslideio::String strProcessing = m_demo_processing ? "blur frame" : "copy frame";
                    ncvslideio::String strTime = ncvslideio::format("time: %4.3f msec", m_timer.getTimeMilli());
                    ncvslideio::String strDevName = ncvslideio::format("OpenCL device: %s", m_oclDevName.c_str());

                    ncvslideio::putText(m, strMode, ncvslideio::Point(0, 20), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(m, strProcessing, ncvslideio::Point(0, 40), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(m, strTime, ncvslideio::Point(0, 60), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(m, strDevName, ncvslideio::Point(0, 80), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);

                    pSurface->Unmap(subResource);

                    break;
                }

                case MODE_GPU_RGBA:
                {
                    // process video frame on GPU
                    ncvslideio::UMat u;

                    ncvslideio::directx::convertFromD3D10Texture2D(pSurface, u);

                    if (m_demo_processing)
                    {
                        // blur D3D10 surface with OpenCV on GPU with OpenCL
                        ncvslideio::blur(u, u, ncvslideio::Size(15, 15));
                    }

                    m_timer.stop();

                    ncvslideio::String strMode = ncvslideio::format("mode: %s", m_modeStr[MODE_GPU_RGBA].c_str());
                    ncvslideio::String strProcessing = m_demo_processing ? "blur frame" : "copy frame";
                    ncvslideio::String strTime = ncvslideio::format("time: %4.3f msec", m_timer.getTimeMilli());
                    ncvslideio::String strDevName = ncvslideio::format("OpenCL device: %s", m_oclDevName.c_str());

                    ncvslideio::putText(u, strMode, ncvslideio::Point(0, 20), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(u, strProcessing, ncvslideio::Point(0, 40), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(u, strTime, ncvslideio::Point(0, 60), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);
                    ncvslideio::putText(u, strDevName, ncvslideio::Point(0, 80), ncvslideio::FONT_HERSHEY_SIMPLEX, 0.8, ncvslideio::Scalar(0, 0, 200), 2);

                    ncvslideio::directx::convertToD3D10Texture2D(u, pSurface);

                    break;
                }

            } // switch

            // traditional DX render pipeline:
            //   BitBlt surface to backBuffer and flip backBuffer to frontBuffer
            m_pD3D10Dev->CopyResource(m_pBackBuffer, pSurface);

            // present the back buffer contents to the display
            // switch the back buffer and the front buffer
            r = m_pD3D10SwapChain->Present(0, 0);
            if (FAILED(r))
            {
                return EXIT_FAILURE;
            }
        } // try

        catch (const ncvslideio::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return EXIT_SUCCESS;
    } // render()


    int cleanup(void)
    {
        SAFE_RELEASE(m_pSurface);
        SAFE_RELEASE(m_pBackBuffer);
        SAFE_RELEASE(m_pD3D10SwapChain);
        SAFE_RELEASE(m_pRenderTarget);
        SAFE_RELEASE(m_pD3D10Dev);
        D3DSample::cleanup();
        return EXIT_SUCCESS;
    } // cleanup()

private:
    ID3D10Device*           m_pD3D10Dev;
    IDXGISwapChain*         m_pD3D10SwapChain;
    ID3D10Texture2D*        m_pBackBuffer;
    ID3D10Texture2D*        m_pSurface;
    ID3D10RenderTargetView* m_pRenderTarget;
    ncvslideio::ocl::Context        m_oclCtx;
    ncvslideio::String              m_oclPlatformName;
    ncvslideio::String              m_oclDevName;
};


// main func
int main(int argc, char** argv)
{
    std::string title = "D3D10 interop sample";
    return d3d_app<D3D10WinApp>(argc, argv, title);
}
