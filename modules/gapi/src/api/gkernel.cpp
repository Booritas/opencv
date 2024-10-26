// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#include "precomp.hpp"
#include <iostream> // cerr
#include <functional> // hash
#include <numeric> // accumulate

#include <ade/util/algorithm.hpp>

#include "logger.hpp"
#include <opencv2/gapi/gkernel.hpp>

#include "api/gbackend_priv.hpp"

// GKernelPackage public implementation ////////////////////////////////////////
void ncvslideio::GKernelPackage::remove(const ncvslideio::gapi::GBackend& backend)
{
    std::vector<std::string> id_deleted_kernels;
    for (const auto& p : m_id_kernels)
    {
        if (p.second.first == backend)
        {
            id_deleted_kernels.push_back(p.first);
        }
    }

    for (const auto& kernel_id : id_deleted_kernels)
    {
        m_id_kernels.erase(kernel_id);
    }
}

void ncvslideio::GKernelPackage::include(const ncvslideio::gapi::GFunctor& functor)
{
    m_id_kernels[functor.id()] = std::make_pair(functor.backend(), functor.impl());
}

void ncvslideio::GKernelPackage::include(const ncvslideio::gapi::GBackend& backend, const std::string& kernel_id)
{
    removeAPI(kernel_id);
    m_id_kernels[kernel_id] = std::make_pair(backend, GKernelImpl{{}, {}});
}

bool ncvslideio::GKernelPackage::includesAPI(const std::string &id) const
{
    return ade::util::contains(m_id_kernels, id);
}

void ncvslideio::GKernelPackage::removeAPI(const std::string &id)
{
    m_id_kernels.erase(id);
}

std::size_t ncvslideio::GKernelPackage::size() const
{
    return m_id_kernels.size();
}

const std::vector<ncvslideio::GTransform> &ncvslideio::GKernelPackage::get_transformations() const
{
    return m_transformations;
}

std::vector<std::string> ncvslideio::GKernelPackage::get_kernel_ids() const
{
    std::vector<std::string> ids;
    for (auto &&id : m_id_kernels)
    {
        ids.emplace_back(id.first);
    }
    return ids;
}

ncvslideio::GKernelPackage ncvslideio::gapi::combine(const ncvslideio::GKernelPackage  &lhs,
                                     const ncvslideio::GKernelPackage  &rhs)
{

        // If there is a collision, prefer RHS to LHS
        // since RHS package has a precedense, start with its copy
        ncvslideio::GKernelPackage result(rhs);
        // now iterate over LHS package and put kernel if and only
        // if there's no such one
        for (const auto& kernel : lhs.m_id_kernels)
        {
            if (!result.includesAPI(kernel.first))
            {
                result.m_id_kernels.emplace(kernel.first, kernel.second);
            }
        }
        for (const auto &transforms : lhs.m_transformations){
            result.m_transformations.push_back(transforms);
        }
        return result;
}

std::pair<ncvslideio::gapi::GBackend, ncvslideio::GKernelImpl>
ncvslideio::GKernelPackage::lookup(const std::string &id) const
{
    auto kernel_it = m_id_kernels.find(id);
    if (kernel_it != m_id_kernels.end())
    {
        return kernel_it->second;
    }
    // If reached here, kernel was not found.
    util::throw_error(std::logic_error("Kernel " + id + " was not found"));
}

std::vector<ncvslideio::gapi::GBackend> ncvslideio::GKernelPackage::backends() const
{
    using kernel_type = std::pair<std::string, std::pair<ncvslideio::gapi::GBackend, ncvslideio::GKernelImpl>>;
    std::unordered_set<ncvslideio::gapi::GBackend> unique_set;
    ade::util::transform(m_id_kernels, std::inserter(unique_set, unique_set.end()),
                                       [](const kernel_type& k) { return k.second.first; });

    return std::vector<ncvslideio::gapi::GBackend>(unique_set.begin(), unique_set.end());
}
