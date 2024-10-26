#!/usr/bin/env python
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests

class stitching_test(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = ncvslideio.Stitcher.create(ncvslideio.Stitcher_PANORAMA)
        (_result, pano) = stitcher.stitch((img1, img2))

        #ncvslideio.imshow("pano", pano)
        #ncvslideio.waitKey()

        self.assertAlmostEqual(pano.shape[0], 685, delta=100, msg="rows: %r" % list(pano.shape))
        self.assertAlmostEqual(pano.shape[1], 1025, delta=100, msg="cols: %r" % list(pano.shape))


class stitching_detail_test(NewOpenCVTests):

    def test_simple(self):
        img = self.get_sample('stitching/a1.png')
        finder= ncvslideio.ORB.create()
        imgFea = ncvslideio.detail.computeImageFeatures2(finder,img)
        self.assertIsNotNone(imgFea)

        # Added Test for PR #21180
        self.assertIsNotNone(imgFea.keypoints)

        matcher = ncvslideio.detail_BestOf2NearestMatcher(False, 0.3)
        self.assertIsNotNone(matcher)
        matcher = ncvslideio.detail_AffineBestOf2NearestMatcher(False, False, 0.3)
        self.assertIsNotNone(matcher)
        matcher = ncvslideio.detail_BestOf2NearestRangeMatcher(2, False, 0.3)
        self.assertIsNotNone(matcher)
        estimator = ncvslideio.detail_AffineBasedEstimator()
        self.assertIsNotNone(estimator)
        estimator = ncvslideio.detail_HomographyBasedEstimator()
        self.assertIsNotNone(estimator)

        adjuster = ncvslideio.detail_BundleAdjusterReproj()
        self.assertIsNotNone(adjuster)
        adjuster = ncvslideio.detail_BundleAdjusterRay()
        self.assertIsNotNone(adjuster)
        adjuster = ncvslideio.detail_BundleAdjusterAffinePartial()
        self.assertIsNotNone(adjuster)
        adjuster = ncvslideio.detail_NoBundleAdjuster()
        self.assertIsNotNone(adjuster)

        compensator=ncvslideio.detail.ExposureCompensator_createDefault(ncvslideio.detail.ExposureCompensator_NO)
        self.assertIsNotNone(compensator)
        compensator=ncvslideio.detail.ExposureCompensator_createDefault(ncvslideio.detail.ExposureCompensator_GAIN)
        self.assertIsNotNone(compensator)
        compensator=ncvslideio.detail.ExposureCompensator_createDefault(ncvslideio.detail.ExposureCompensator_GAIN_BLOCKS)
        self.assertIsNotNone(compensator)

        seam_finder = ncvslideio.detail.SeamFinder_createDefault(ncvslideio.detail.SeamFinder_NO)
        self.assertIsNotNone(seam_finder)
        seam_finder = ncvslideio.detail.SeamFinder_createDefault(ncvslideio.detail.SeamFinder_NO)
        self.assertIsNotNone(seam_finder)
        seam_finder = ncvslideio.detail.SeamFinder_createDefault(ncvslideio.detail.SeamFinder_VORONOI_SEAM)
        self.assertIsNotNone(seam_finder)

        seam_finder = ncvslideio.detail_GraphCutSeamFinder("COST_COLOR")
        self.assertIsNotNone(seam_finder)
        seam_finder = ncvslideio.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        self.assertIsNotNone(seam_finder)
        seam_finder = ncvslideio.detail_DpSeamFinder("COLOR")
        self.assertIsNotNone(seam_finder)
        seam_finder = ncvslideio.detail_DpSeamFinder("COLOR_GRAD")
        self.assertIsNotNone(seam_finder)

        blender = ncvslideio.detail.Blender_createDefault(ncvslideio.detail.Blender_NO)
        self.assertIsNotNone(blender)
        blender = ncvslideio.detail.Blender_createDefault(ncvslideio.detail.Blender_FEATHER)
        self.assertIsNotNone(blender)
        blender = ncvslideio.detail.Blender_createDefault(ncvslideio.detail.Blender_MULTI_BAND)
        self.assertIsNotNone(blender)

        timelapser = ncvslideio.detail.Timelapser_createDefault(ncvslideio.detail.Timelapser_AS_IS);
        self.assertIsNotNone(timelapser)
        timelapser = ncvslideio.detail.Timelapser_createDefault(ncvslideio.detail.Timelapser_CROP);
        self.assertIsNotNone(timelapser)


class stitching_compose_panorama_test_no_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = ncvslideio.Stitcher.create(ncvslideio.Stitcher_PANORAMA)

        stitcher.estimateTransform((img1, img2))

        result, _ = stitcher.composePanorama()

        assert result == 0


class stitching_compose_panorama_args(NewOpenCVTests):

    def test_simple(self):

        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        stitcher = ncvslideio.Stitcher.create(ncvslideio.Stitcher_PANORAMA)

        stitcher.estimateTransform((img1, img2))
        result, _ = stitcher.composePanorama((img1, img2))

        assert result == 0


class stitching_matches_info_test(NewOpenCVTests):

    def test_simple(self):
        finder = ncvslideio.ORB.create()
        img1 = self.get_sample('stitching/a1.png')
        img2 = self.get_sample('stitching/a2.png')

        img_feat1 = ncvslideio.detail.computeImageFeatures2(finder, img1)
        img_feat2 = ncvslideio.detail.computeImageFeatures2(finder, img2)

        matcher = ncvslideio.detail.BestOf2NearestMatcher_create()
        matches_info = matcher.apply(img_feat1, img_feat2)

        self.assertIsNotNone(matches_info.matches)
        self.assertIsNotNone(matches_info.inliers_mask)

class stitching_range_matcher_test(NewOpenCVTests):

    def test_simple(self):
        images = [
            self.get_sample('stitching/a1.png'),
            self.get_sample('stitching/a2.png'),
            self.get_sample('stitching/a3.png')
        ]

        orb = ncvslideio.ORB_create()

        features = [ncvslideio.detail.computeImageFeatures2(orb, img) for img in images]

        matcher = ncvslideio.detail_BestOf2NearestRangeMatcher(range_width=1)
        matches = matcher.apply2(features)

        # matches[1] is image 0 and image 1, should have non-zero confidence
        self.assertNotEqual(matches[1].confidence, 0)

        # matches[2] is image 0 and image 2, should have zero confidence due to range_width=1
        self.assertEqual(matches[2].confidence, 0)


class stitching_seam_finder_graph_cuts(NewOpenCVTests):

    def test_simple(self):
        images = [
            self.get_sample('stitching/a1.png'),
            self.get_sample('stitching/a2.png'),
            self.get_sample('stitching/a3.png')
        ]

        images = [ncvslideio.resize(img, [100, 100]) for img in images]

        finder = ncvslideio.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
        masks = [ncvslideio.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8)) for img in images]
        images_f = [img.astype(np.float32) for img in images]
        masks_warped = finder.find(images_f, [(0, 0), (75, 0), (150, 0)], masks)

        self.assertIsNotNone(masks_warped)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
