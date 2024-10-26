// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

function generateTestFrame(width, height) {
  let w = width || 200;
  let h = height || 200;
  let img = new ncvslideio.Mat(h, w, ncvslideio.CV_8UC1, new ncvslideio.Scalar(0, 0, 0, 0));
  let s = new ncvslideio.Scalar(255, 255, 255, 255);
  let s128 = new ncvslideio.Scalar(128, 128, 128, 128);
  let rect = new ncvslideio.Rect(w / 4, h / 4, w / 2, h / 2);
  img.roi(rect).setTo(s);
  img.roi(new ncvslideio.Rect(w / 2 - w / 8, h / 2 - h / 8, w / 4, h / 4)).setTo(s128);
  ncvslideio.rectangle(img, new ncvslideio.Point(w / 8, h / 8), new ncvslideio.Point(w - w / 8, h - h / 8), s, 5);
  ncvslideio.rectangle(img, new ncvslideio.Point(w / 5, h / 5), new ncvslideio.Point(w - w / 5, h - h / 5), s128, 3);
  ncvslideio.line(img, new ncvslideio.Point(-w, 0), new ncvslideio.Point(w / 2, h / 2), s128, 5);
  ncvslideio.line(img, new ncvslideio.Point(2*w, 0), new ncvslideio.Point(w / 2, h / 2), s, 5);
  return img;
}

QUnit.module('Features2D', {});
QUnit.test('Detectors', function(assert) {
  let image = generateTestFrame();

  let kp = new ncvslideio.KeyPointVector();

  let orb = new ncvslideio.ORB();
  orb.detect(image, kp);
  assert.equal(kp.size(), 67, 'ORB');

  let mser = new ncvslideio.MSER();
  mser.detect(image, kp);
  assert.equal(kp.size(), 7, 'MSER');

  let brisk = new ncvslideio.BRISK();
  brisk.detect(image, kp);
  assert.equal(kp.size(), 191, 'BRISK');

  let ffd = new ncvslideio.FastFeatureDetector();
  ffd.detect(image, kp);
  assert.equal(kp.size(), 12, 'FastFeatureDetector');

  let afd = new ncvslideio.AgastFeatureDetector();
  afd.detect(image, kp);
  assert.equal(kp.size(), 67, 'AgastFeatureDetector');

  let gftt = new ncvslideio.GFTTDetector();
  gftt.detect(image, kp);
  assert.equal(kp.size(), 168, 'GFTTDetector');

  let kaze = new ncvslideio.KAZE();
  kaze.detect(image, kp);
  assert.equal(kp.size(), 159, 'KAZE');

  let akaze = new ncvslideio.AKAZE();
  akaze.detect(image, kp);
  assert.equal(kp.size(), 53, 'AKAZE');
});

QUnit.test('SimpleBlobDetector', function(assert) {
  let image = generateTestFrame();

  let kp = new ncvslideio.KeyPointVector();
  let sbd = new ncvslideio.SimpleBlobDetector();
  sbd.detect(image, kp);
  assert.equal(kp.size(), 0);
});

QUnit.test('BFMatcher', function(assert) {
  // Generate key points.
  let image = generateTestFrame();

  let kp = new ncvslideio.KeyPointVector();
  let descriptors = new ncvslideio.Mat();
  let orb = new ncvslideio.ORB();
  orb.detectAndCompute(image, new ncvslideio.Mat(), kp, descriptors);

  assert.equal(kp.size(), 67);

  // Run a matcher.
  let dm = new ncvslideio.DMatchVector();
  let matcher = new ncvslideio.BFMatcher();
  matcher.match(descriptors, descriptors, dm);

  assert.equal(dm.size(), 67);
});

QUnit.test('Drawing', function(assert) {
  // Generate key points.
  let image = generateTestFrame();

  let kp = new ncvslideio.KeyPointVector();
  let descriptors = new ncvslideio.Mat();
  let orb = new ncvslideio.ORB();
  orb.detectAndCompute(image, new ncvslideio.Mat(), kp, descriptors);
  assert.equal(kp.size(), 67);

  let dst = new ncvslideio.Mat();
  ncvslideio.drawKeypoints(image, kp, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, image.cols);

  // Run a matcher.
  let dm = new ncvslideio.DMatchVector();
  let matcher = new ncvslideio.BFMatcher();
  matcher.match(descriptors, descriptors, dm);
  assert.equal(dm.size(), 67);

  ncvslideio.drawMatches(image, kp, image, kp, dm, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, 2 * image.cols);

  dm = new ncvslideio.DMatchVectorVector();
  matcher.knnMatch(descriptors, descriptors, dm, 2);
  assert.equal(dm.size(), 67);
  ncvslideio.drawMatchesKnn(image, kp, image, kp, dm, dst);
  assert.equal(dst.rows, image.rows);
  assert.equal(dst.cols, 2 * image.cols);
});
