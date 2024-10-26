// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_FILESYSTEM_HPP
#define OPENCV_UTILS_FILESYSTEM_HPP

namespace ncvslideio { namespace utils { namespace fs {


CV_EXPORTS bool exists(const ncvslideio::String& path);
CV_EXPORTS bool isDirectory(const ncvslideio::String& path);

CV_EXPORTS void remove_all(const ncvslideio::String& path);


CV_EXPORTS ncvslideio::String getcwd();

/** @brief Converts path p to a canonical absolute path
 * Symlinks are processed if there is support for them on running platform.
 *
 * @param path input path. Target file/directory should exist.
 */
CV_EXPORTS ncvslideio::String canonical(const ncvslideio::String& path);

/** Join path components */
CV_EXPORTS ncvslideio::String join(const ncvslideio::String& base, const ncvslideio::String& path);

/** Get parent directory */
CV_EXPORTS ncvslideio::String getParent(const ncvslideio::String &path);
CV_EXPORTS std::wstring getParent(const std::wstring& path);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * Result entries are prefixed by base directory path.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to disable filtering and return all results
 * @param[out] result result of globing.
 * @param recursive scan nested directories too
 * @param includeDirectories include directories into results list
 */
CV_EXPORTS void glob(const ncvslideio::String& directory, const ncvslideio::String& pattern,
        CV_OUT std::vector<ncvslideio::String>& result,
        bool recursive = false, bool includeDirectories = false);

/**
 * Generate a list of all files that match the globbing pattern.
 *
 * @param directory base directory
 * @param pattern filter pattern (based on '*'/'?' symbols). Use empty string to disable filtering and return all results
 * @param[out] result globbing result with relative paths from base directory
 * @param recursive scan nested directories too
 * @param includeDirectories include directories into results list
 */
CV_EXPORTS void glob_relative(const ncvslideio::String& directory, const ncvslideio::String& pattern,
        CV_OUT std::vector<ncvslideio::String>& result,
        bool recursive = false, bool includeDirectories = false);


CV_EXPORTS bool createDirectory(const ncvslideio::String& path);
CV_EXPORTS bool createDirectories(const ncvslideio::String& path);

#if defined(__OPENCV_BUILD) || defined(BUILD_PLUGIN)
// TODO
//CV_EXPORTS ncvslideio::String getTempDirectory();

/**
 * @brief Returns directory to store OpenCV cache files
 * Create sub-directory in common OpenCV cache directory if it doesn't exist.
 * @param sub_directory_name name of sub-directory. NULL or "" value asks to return root cache directory.
 * @param configuration_name optional name of configuration parameter name which overrides default behavior.
 * @return Path to cache directory. Returns empty string if cache directories support is not available. Returns "disabled" if cache disabled by user.
 */
CV_EXPORTS ncvslideio::String getCacheDirectory(const char* sub_directory_name, const char* configuration_name = NULL);

#endif

}}} // namespace

#endif // OPENCV_UTILS_FILESYSTEM_HPP
