// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Dummy allocator implementation copied from the default OpenCV allocator with some simplifications
struct DummyAllocator: public ncvslideio::MatAllocator
{
public:
    DummyAllocator() {};
    ~DummyAllocator() {};

    ncvslideio::UMatData* allocate(int dims, const int* sizes, int type,
                    void* data0, size_t* step, ncvslideio::AccessFlag flags,
                    ncvslideio::UMatUsageFlags usageFlags) const
    {
        CV_UNUSED(flags);
        CV_UNUSED(usageFlags);

        size_t total = CV_ELEM_SIZE(type);
        for( int i = dims-1; i >= 0; i-- )
        {
            if( step )
            {
                if( data0 && step[i] != CV_AUTOSTEP )
                {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                }
                else
                    step[i] = total;
            }
            total *= sizes[i];
        }

        uchar* data = nullptr;
        if (data0)
        {
            data = (uchar*)data0;
        }
        else
        {
            data = new uchar[total];
            DummyAllocator::allocatedBytes += total;
            DummyAllocator::allocations++;
        }
        ncvslideio::UMatData* u = new ncvslideio::UMatData(this);
        u->data = u->origdata = data;
        u->size = total;
        if(data0)
            u->flags |= ncvslideio::UMatData::USER_ALLOCATED;

        return u;
    }

    bool allocate(ncvslideio::UMatData* u, ncvslideio::AccessFlag accessFlags, ncvslideio::UMatUsageFlags usageFlags) const
    {
        CV_UNUSED(accessFlags);
        CV_UNUSED(usageFlags);

        if(!u) return false;
        return true;
    }

    void deallocate(ncvslideio::UMatData* u) const
    {
        if(!u)
            return;

        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        if( !(u->flags & ncvslideio::UMatData::USER_ALLOCATED) )
        {
            delete[] u->origdata;
            DummyAllocator::deallocations++;
            u->origdata = 0;
        }
        delete u;
    }

    static size_t allocatedBytes;
    static int allocations;
    static int deallocations;
};

size_t DummyAllocator::allocatedBytes = 0;
int  DummyAllocator::allocations = 0;
int  DummyAllocator::deallocations = 0;

ncvslideio::MatAllocator* getDummyAllocator()
{
    static ncvslideio::MatAllocator* allocator = new DummyAllocator;
    return allocator;
}

struct AllocatorTest : public testing::Test {
    void SetUp() override {
        ncvslideio::MatAllocator* allocator = getDummyAllocator();
        EXPECT_TRUE(allocator != nullptr);
        ncvslideio::Mat::setDefaultAllocator(allocator);
    }

    void TearDown() override {
        ncvslideio::Mat::setDefaultAllocator(ncvslideio::Mat::getStdAllocator());
    }
};

TEST_F(AllocatorTest, DummyAllocator)
{
    ncvslideio::MatAllocator* dummy = getDummyAllocator();

    DummyAllocator::allocatedBytes = 0;
    DummyAllocator::allocations = 0;
    DummyAllocator::deallocations = 0;

    {
        ncvslideio::Mat src1 = ncvslideio::Mat::ones (16, 16, CV_8UC1);
        EXPECT_TRUE(!src1.empty());
        EXPECT_EQ(src1.allocator, dummy);

        ncvslideio::Mat src1_roi = src1(ncvslideio::Rect(2,2,8,8));
        EXPECT_EQ(src1_roi.allocator, dummy);

        ncvslideio::MatAllocator* standard = ncvslideio::Mat::getStdAllocator();
        ncvslideio::Mat::setDefaultAllocator(standard);
        ncvslideio::Mat src2 = ncvslideio::Mat::ones (16, 16, CV_8UC1);
        EXPECT_TRUE(!src2.empty());
        EXPECT_EQ(src2.allocator, standard);

        src1.create(32, 32, CV_8UC1);
        EXPECT_EQ(src1.allocator, dummy);
    }

    size_t expect_allocated = 16*16*sizeof(uchar) + 32*32*sizeof(uchar);
    EXPECT_EQ(expect_allocated, DummyAllocator::allocatedBytes);

    // ROI should not trigger extra allocations
    EXPECT_EQ(2, DummyAllocator::allocations);
    EXPECT_EQ(2, DummyAllocator::deallocations);
}

}} // namespace
