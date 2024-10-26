/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_OBJDETECT_DBT_HPP
#define OPENCV_OBJDETECT_DBT_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace ncvslideio
{

//! @addtogroup objdetect_cascade_classifier
//! @{

class CV_EXPORTS DetectionBasedTracker
{
    public:
        struct CV_EXPORTS Parameters
        {
            int maxTrackLifetime;
            int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0

            Parameters();
        };

        class IDetector
        {
            public:
                IDetector():
                    minObjSize(96, 96),
                    maxObjSize(INT_MAX, INT_MAX),
                    minNeighbours(2),
                    scaleFactor(1.1f)
                {}

                virtual void detect(const ncvslideio::Mat& image, std::vector<ncvslideio::Rect>& objects) = 0;

                void setMinObjectSize(const ncvslideio::Size& min)
                {
                    minObjSize = min;
                }
                void setMaxObjectSize(const ncvslideio::Size& max)
                {
                    maxObjSize = max;
                }
                ncvslideio::Size getMinObjectSize() const
                {
                    return minObjSize;
                }
                ncvslideio::Size getMaxObjectSize() const
                {
                    return maxObjSize;
                }
                float getScaleFactor()
                {
                    return scaleFactor;
                }
                void setScaleFactor(float value)
                {
                    scaleFactor = value;
                }
                int getMinNeighbours()
                {
                    return minNeighbours;
                }
                void setMinNeighbours(int value)
                {
                    minNeighbours = value;
                }
                virtual ~IDetector() {}

            protected:
                ncvslideio::Size minObjSize;
                ncvslideio::Size maxObjSize;
                int minNeighbours;
                float scaleFactor;
        };

        DetectionBasedTracker(ncvslideio::Ptr<IDetector> mainDetector, ncvslideio::Ptr<IDetector> trackingDetector, const Parameters& params);
        virtual ~DetectionBasedTracker();

        virtual bool run();
        virtual void stop();
        virtual void resetTracking();

        virtual void process(const ncvslideio::Mat& imageGray);

        bool setParameters(const Parameters& params);
        const Parameters& getParameters() const;


        typedef std::pair<ncvslideio::Rect, int> Object;
        virtual void getObjects(std::vector<ncvslideio::Rect>& result) const;
        virtual void getObjects(std::vector<Object>& result) const;

        enum ObjectStatus
        {
            DETECTED_NOT_SHOWN_YET,
            DETECTED,
            DETECTED_TEMPORARY_LOST,
            WRONG_OBJECT
        };
        struct ExtObject
        {
            int id;
            ncvslideio::Rect location;
            ObjectStatus status;
            ExtObject(int _id, ncvslideio::Rect _location, ObjectStatus _status)
                :id(_id), location(_location), status(_status)
            {
            }
        };
        virtual void getObjects(std::vector<ExtObject>& result) const;


        virtual int addObject(const ncvslideio::Rect& location); //returns id of the new object

    protected:
        class SeparateDetectionWork;
        ncvslideio::Ptr<SeparateDetectionWork> separateDetectionWork;
        friend void* workcycleObjectDetectorFunction(void* p);

        struct InnerParameters
        {
            int numLastPositionsToTrack;
            int numStepsToWaitBeforeFirstShow;
            int numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown;
            int numStepsToShowWithoutDetecting;

            float coeffTrackingWindowSize;
            float coeffObjectSizeToTrack;
            float coeffObjectSpeedUsingInPrediction;

            InnerParameters();
        };
        Parameters parameters;
        InnerParameters innerParameters;

        struct TrackedObject
        {
            typedef std::vector<ncvslideio::Rect> PositionsVector;

            PositionsVector lastPositions;

            int numDetectedFrames;
            int numFramesNotDetected;
            int id;

            TrackedObject(const ncvslideio::Rect& rect):numDetectedFrames(1), numFramesNotDetected(0)
            {
                lastPositions.push_back(rect);
                id=getNextId();
            }

            static int getNextId()
            {
                static int _id=0;
                return _id++;
            }
        };

        int numTrackedSteps;
        std::vector<TrackedObject> trackedObjects;

        std::vector<float> weightsPositionsSmoothing;
        std::vector<float> weightsSizesSmoothing;

        ncvslideio::Ptr<IDetector> cascadeForTracking;

        void updateTrackedObjects(const std::vector<ncvslideio::Rect>& detectedObjects);
        ncvslideio::Rect calcTrackedObjectPositionToShow(int i) const;
        ncvslideio::Rect calcTrackedObjectPositionToShow(int i, ObjectStatus& status) const;
        void detectInRegion(const ncvslideio::Mat& img, const ncvslideio::Rect& r, std::vector<ncvslideio::Rect>& detectedObjectsInRegions);
};

//! @}

} //end of ncvslideio namespace

#endif
