//
//  DetectionBasedTracker.mm
//
//  Created by Giles Payne on 2020/04/05.
//

#import "DetectionBasedTracker.h"
#import "Mat.h"
#import "Rect2i.h"
#import "CVObjcUtil.h"

class CascadeDetectorAdapter: public ncvslideio::DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(ncvslideio::Ptr<ncvslideio::CascadeClassifier> detector):IDetector(), Detector(detector) {}

    void detect(const ncvslideio::Mat &Image, std::vector<ncvslideio::Rect> &objects)
    {
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }

    virtual ~CascadeDetectorAdapter() {}

private:
    CascadeDetectorAdapter();
    ncvslideio::Ptr<ncvslideio::CascadeClassifier> Detector;
};


struct DetectorAgregator
{
    ncvslideio::Ptr<CascadeDetectorAdapter> mainDetector;
    ncvslideio::Ptr<CascadeDetectorAdapter> trackingDetector;
    ncvslideio::Ptr<ncvslideio::DetectionBasedTracker> tracker;
    DetectorAgregator(ncvslideio::Ptr<CascadeDetectorAdapter>& _mainDetector, ncvslideio::Ptr<CascadeDetectorAdapter>& _trackingDetector):mainDetector(_mainDetector), trackingDetector(_trackingDetector) {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);
        ncvslideio::DetectionBasedTracker::Parameters DetectorParams;
        tracker = ncvslideio::makePtr<ncvslideio::DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

@implementation DetectionBasedTracker {
    DetectorAgregator* agregator;
}

- (instancetype)initWithCascadeName:(NSString*)cascadeName minFaceSize:(int)faceSize {
    self = [super init];
    if (self) {
        auto mainDetector = ncvslideio::makePtr<CascadeDetectorAdapter>(ncvslideio::makePtr<ncvslideio::CascadeClassifier>(cascadeName.UTF8String));
        auto trackingDetector = ncvslideio::makePtr<CascadeDetectorAdapter>(
            ncvslideio::makePtr<ncvslideio::CascadeClassifier>(cascadeName.UTF8String));
        agregator = new DetectorAgregator(mainDetector, trackingDetector);
        if (faceSize > 0) {
            agregator->mainDetector->setMinObjectSize(ncvslideio::Size(faceSize, faceSize));
        }
    }
    return self;
}

- (void)dealloc
{
    delete agregator;
}

- (void)start {
    agregator->tracker->run();
}

- (void)stop {
    agregator->tracker->stop();
}

- (void)setFaceSize:(int)size {
    agregator->mainDetector->setMinObjectSize(ncvslideio::Size(size, size));
}

- (void)detect:(Mat*)imageGray faces:(NSMutableArray<Rect2i*>*)faces {
    std::vector<ncvslideio::Rect> rectFaces;
    agregator->tracker->process(*((ncvslideio::Mat*)imageGray.nativePtr));
    agregator->tracker->getObjects(rectFaces);
    CV2OBJC(ncvslideio::Rect, Rect2i, rectFaces, faces);
}

@end
