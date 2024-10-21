// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "opencv_data_config.hpp"

#include <vector>
#include <fstream>

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/filesystem.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include "opencv2/core/utils/filesystem.private.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef small
#undef min
#undef max
#undef abs
#elif defined(__linux__)
#include <dlfcn.h>  // requires -ldl
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#include <dlfcn.h>
#endif
#endif

namespace ncvslideio { namespace utils {

static ncvslideio::Ptr< std::vector<ncvslideio::String> > g_data_search_path;
static ncvslideio::Ptr< std::vector<ncvslideio::String> > g_data_search_subdir;

static std::vector<ncvslideio::String>& _getDataSearchPath()
{
    if (g_data_search_path.empty())
        g_data_search_path.reset(new std::vector<ncvslideio::String>());
    return *(g_data_search_path.get());
}

static std::vector<ncvslideio::String>& _getDataSearchSubDirectory()
{
    if (g_data_search_subdir.empty())
    {
        g_data_search_subdir.reset(new std::vector<ncvslideio::String>());
        g_data_search_subdir->push_back("data");
        g_data_search_subdir->push_back("");
    }
    return *(g_data_search_subdir.get());
}


CV_EXPORTS void addDataSearchPath(const ncvslideio::String& path)
{
    if (utils::fs::isDirectory(path))
        _getDataSearchPath().push_back(path);
}
CV_EXPORTS void addDataSearchSubDirectory(const ncvslideio::String& subdir)
{
    _getDataSearchSubDirectory().push_back(subdir);
}

#if OPENCV_HAVE_FILESYSTEM_SUPPORT
static bool isPathSep(char c)
{
    return c == '/' || c == '\\';
}
static bool isSubDirectory_(const ncvslideio::String& base_path, const ncvslideio::String& path)
{
    size_t N = base_path.size();
    if (N == 0)
        return false;
    if (isPathSep(base_path[N - 1]))
        N--;
    if (path.size() < N)
        return false;
    for (size_t i = 0; i < N; i++)
    {
        if (path[i] == base_path[i])
            continue;
        if (isPathSep(path[i]) && isPathSep(base_path[i]))
            continue;
        return false;
    }
    size_t M = path.size();
    if (M > N)
    {
        if (!isPathSep(path[N]))
            return false;
    }
    return true;
}

static bool isSubDirectory(const ncvslideio::String& base_path, const ncvslideio::String& path)
{
    bool res = isSubDirectory_(base_path, path);
    CV_LOG_VERBOSE(NULL, 0, "isSubDirectory(): base: " << base_path << "  path: " << path << "  => result: " << (res ? "TRUE" : "FALSE"));
    return res;
}
#endif //OPENCV_HAVE_FILESYSTEM_SUPPORT

static ncvslideio::String getModuleLocation(const void* addr)
{
    CV_UNUSED(addr);
#ifdef _WIN32
    HMODULE m = 0;
#if _WIN32_WINNT >= 0x0501 && (!defined(WINAPI_FAMILY) || (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP))
    ::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCTSTR>(addr),
        &m);
#endif
    if (m)
    {
        TCHAR path[MAX_PATH];
        const size_t path_size = sizeof(path) / sizeof(*path);
        size_t sz = GetModuleFileName(m, path, path_size);
        if (sz > 0 && sz < path_size)
        {
            path[sz] = TCHAR('\0');
#ifdef _UNICODE
            char char_path[MAX_PATH];
            size_t copied = wcstombs(char_path, path, MAX_PATH);
            CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
            return ncvslideio::String(char_path);
#else
            return ncvslideio::String(path);
#endif
        }
    }
#elif defined(__linux__)
    Dl_info info;
    if (0 != dladdr(addr, &info))
    {
        return ncvslideio::String(info.dli_fname);
    }
#elif defined(__APPLE__)
# if TARGET_OS_MAC
    Dl_info info;
    if (0 != dladdr(addr, &info))
    {
        return ncvslideio::String(info.dli_fname);
    }
# endif
#else
    // not supported, skip
#endif
    return ncvslideio::String();
}

bool getBinLocation(std::string& dst)
{
    dst = getModuleLocation((void*)getModuleLocation); // using code address, doesn't work with static linkage!
    return !dst.empty();
}

#ifdef _WIN32
bool getBinLocation(std::wstring& dst)
{
    void* addr = (void*)getModuleLocation; // using code address, doesn't work with static linkage!
    HMODULE m = 0;
#if _WIN32_WINNT >= 0x0501 && (!defined(WINAPI_FAMILY) || (WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP))
    ::GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCTSTR>(addr),
        &m);
#endif
    if (m)
    {
        wchar_t path[4096];
        const size_t path_size = sizeof(path)/sizeof(*path);
        size_t sz = GetModuleFileNameW(m, path, path_size);
        if (sz > 0 && sz < path_size)
        {
            path[sz] = '\0';
            dst.assign(path, sz);
            return true;
        }
    }
    return false;
}
#endif

ncvslideio::String findDataFile(const ncvslideio::String& relative_path,
                        const char* configuration_parameter,
                        const std::vector<String>* search_paths,
                        const std::vector<String>* subdir_paths)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    configuration_parameter = configuration_parameter ? configuration_parameter : "OPENCV_DATA_PATH";
    CV_LOG_DEBUG(NULL, ncvslideio::format("utils::findDataFile('%s', %s)", relative_path.c_str(), configuration_parameter));

#define TRY_FILE_WITH_PREFIX(prefix) \
{ \
    ncvslideio::String path = utils::fs::join(prefix, relative_path); \
    CV_LOG_DEBUG(NULL, ncvslideio::format("... Line %d: trying open '%s'", __LINE__, path.c_str())); \
    FILE* f = fopen(path.c_str(), "rb"); \
    if(f) { \
        fclose(f); \
        return path; \
    } \
}


    // Step 0: check current directory or absolute path at first
    TRY_FILE_WITH_PREFIX("");


    // Step 1
    const std::vector<ncvslideio::String>& search_path = search_paths ? *search_paths : _getDataSearchPath();
    for(size_t i = search_path.size(); i > 0; i--)
    {
        const ncvslideio::String& prefix = search_path[i - 1];
        TRY_FILE_WITH_PREFIX(prefix);
    }

    const std::vector<ncvslideio::String>& search_subdir = subdir_paths ? *subdir_paths : _getDataSearchSubDirectory();


    // Step 2
    const ncvslideio::String configuration_parameter_s(configuration_parameter ? configuration_parameter : "");
    const ncvslideio::utils::Paths& search_hint = configuration_parameter_s.empty() ? ncvslideio::utils::Paths()
                                          : getConfigurationParameterPaths((configuration_parameter_s + "_HINT").c_str());
    for (size_t k = 0; k < search_hint.size(); k++)
    {
        ncvslideio::String datapath = search_hint[k];
        if (datapath.empty())
            continue;
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): trying " << configuration_parameter << "_HINT=" << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const ncvslideio::String& subdir = search_subdir[i - 1];
                ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
        else
        {
            CV_LOG_WARNING(NULL, configuration_parameter << "_HINT is specified but it is not a directory: " << datapath);
        }
    }


    // Step 3
    const ncvslideio::utils::Paths& override_paths = configuration_parameter_s.empty() ? ncvslideio::utils::Paths()
                                           : getConfigurationParameterPaths(configuration_parameter);
    for (size_t k = 0; k < override_paths.size(); k++)
    {
        ncvslideio::String datapath = override_paths[k];
        if (datapath.empty())
            continue;
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): trying " << configuration_parameter << "=" << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const ncvslideio::String& subdir = search_subdir[i - 1];
                ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
        else
        {
            CV_LOG_WARNING(NULL, configuration_parameter << " is specified but it is not a directory: " << datapath);
        }
    }
    if (!override_paths.empty())
    {
        CV_LOG_INFO(NULL, "utils::findDataFile(): can't find data file via " << configuration_parameter << " configuration override: " << relative_path);
        return ncvslideio::String();
    }


    // Steps: 4, 5, 6
    ncvslideio::String cwd = utils::fs::getcwd();
    ncvslideio::String build_dir(OPENCV_BUILD_DIR);
    bool has_tested_build_directory = false;
    if (isSubDirectory(build_dir, cwd) || isSubDirectory(utils::fs::canonical(build_dir), utils::fs::canonical(cwd)))
    {
        CV_LOG_DEBUG(NULL, "utils::findDataFile(): the current directory is build sub-directory: " << cwd);
        const char* build_subdirs[] = { OPENCV_DATA_BUILD_DIR_SEARCH_PATHS };
        for (size_t k = 0; k < sizeof(build_subdirs)/sizeof(build_subdirs[0]); k++)
        {
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): <build>/" << build_subdirs[k]);
            ncvslideio::String datapath = utils::fs::join(build_dir, build_subdirs[k]);
            if (utils::fs::isDirectory(datapath))
            {
                for(size_t i = search_subdir.size(); i > 0; i--)
                {
                    const ncvslideio::String& subdir = search_subdir[i - 1];
                    ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                    TRY_FILE_WITH_PREFIX(prefix);
                }
            }
        }
        has_tested_build_directory = true;
    }

    ncvslideio::String source_dir;
    ncvslideio::String try_source_dir = cwd;
    for (int levels = 0; levels < 3; ++levels)
    {
        if (utils::fs::exists(utils::fs::join(try_source_dir, "modules/core/include/opencv2/core/version.hpp")))
        {
            source_dir = try_source_dir;
            break;
        }
        try_source_dir = utils::fs::join(try_source_dir, "/..");
    }
    if (!source_dir.empty())
    {
        CV_LOG_DEBUG(NULL, "utils::findDataFile(): the current directory is source sub-directory: " << source_dir);
        CV_LOG_DEBUG(NULL, "utils::findDataFile(): <source>" << source_dir);
        ncvslideio::String datapath = source_dir;
        if (utils::fs::isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const ncvslideio::String& subdir = search_subdir[i - 1];
                ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }

    ncvslideio::String module_path;
    if (getBinLocation(module_path))
    {
        CV_LOG_DEBUG(NULL, "Detected module path: '" << module_path << '\'');
    }
    else
    {
        CV_LOG_INFO(NULL, "Can't detect module binaries location");
    }

    if (!has_tested_build_directory &&
        (isSubDirectory(build_dir, module_path) || isSubDirectory(utils::fs::canonical(build_dir), utils::fs::canonical(module_path)))
    )
    {
        CV_LOG_DEBUG(NULL, "utils::findDataFile(): the binary module directory is build sub-directory: " << module_path);
        const char* build_subdirs[] = { OPENCV_DATA_BUILD_DIR_SEARCH_PATHS };
        for (size_t k = 0; k < sizeof(build_subdirs)/sizeof(build_subdirs[0]); k++)
        {
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): <build>/" << build_subdirs[k]);
            ncvslideio::String datapath = utils::fs::join(build_dir, build_subdirs[k]);
            if (utils::fs::isDirectory(datapath))
            {
                for(size_t i = search_subdir.size(); i > 0; i--)
                {
                    const ncvslideio::String& subdir = search_subdir[i - 1];
                    ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                    TRY_FILE_WITH_PREFIX(prefix);
                }
            }
        }
    }

#if defined OPENCV_INSTALL_DATA_DIR_RELATIVE
    if (!module_path.empty())  // require module path
    {
        size_t pos = module_path.rfind('/');
        if (pos == ncvslideio::String::npos)
            pos = module_path.rfind('\\');
        ncvslideio::String module_dir = (pos == ncvslideio::String::npos) ? module_path : module_path.substr(0, pos);
        const char* install_subdirs[] = { OPENCV_INSTALL_DATA_DIR_RELATIVE };
        for (size_t k = 0; k < sizeof(install_subdirs)/sizeof(install_subdirs[0]); k++)
        {
            ncvslideio::String datapath = utils::fs::join(module_dir, install_subdirs[k]);
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): trying install path (from binary path): " << datapath);
            if (utils::fs::isDirectory(datapath))
            {
                for(size_t i = search_subdir.size(); i > 0; i--)
                {
                    const ncvslideio::String& subdir = search_subdir[i - 1];
                    ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                    TRY_FILE_WITH_PREFIX(prefix);
                }
            }
            else
            {
                CV_LOG_DEBUG(NULL, "utils::findDataFile(): ... skip, not a valid directory: " << datapath);
            }
        }
    }
#endif

#if defined OPENCV_INSTALL_PREFIX && defined OPENCV_DATA_INSTALL_PATH
    ncvslideio::String install_dir(OPENCV_INSTALL_PREFIX);
    // use core/world module path and verify that library is running from installation directory
    // It is necessary to avoid touching of unrelated common /usr/local path
    if (module_path.empty()) // can't determine
        module_path = install_dir;
    if (isSubDirectory(install_dir, module_path) || isSubDirectory(utils::fs::canonical(install_dir), utils::fs::canonical(module_path)))
    {
        ncvslideio::String datapath = utils::fs::join(install_dir, OPENCV_DATA_INSTALL_PATH);
        if (utils::fs::isDirectory(datapath))
        {
            CV_LOG_DEBUG(NULL, "utils::findDataFile(): trying install path: " << datapath);
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const ncvslideio::String& subdir = search_subdir[i - 1];
                ncvslideio::String prefix = utils::fs::join(datapath, subdir);
                TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#endif

    return ncvslideio::String();  // not found
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(relative_path);
    CV_UNUSED(configuration_parameter);
    CV_UNUSED(search_paths);
    CV_UNUSED(subdir_paths);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif // OPENCV_HAVE_FILESYSTEM_SUPPORT
}

ncvslideio::String findDataFile(const ncvslideio::String& relative_path, bool required, const char* configuration_parameter)
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_LOG_DEBUG(NULL, ncvslideio::format("ncvslideio::utils::findDataFile('%s', %s, %s)",
                                  relative_path.c_str(), required ? "true" : "false",
                                  configuration_parameter ? configuration_parameter : "NULL"));
    ncvslideio::String result = ncvslideio::utils::findDataFile(relative_path,
                                                configuration_parameter,
                                                NULL,
                                                NULL);
    if (result.empty() && required)
        CV_Error(ncvslideio::Error::StsError, ncvslideio::format("OpenCV: Can't find required data file: %s", relative_path.c_str()));
    return result;
#else // OPENCV_HAVE_FILESYSTEM_SUPPORT
    CV_UNUSED(relative_path);
    CV_UNUSED(required);
    CV_UNUSED(configuration_parameter);
    CV_Error(Error::StsNotImplemented, "File system support is disabled in this OpenCV build!");
#endif // OPENCV_HAVE_FILESYSTEM_SUPPORT
}

}} // namespace
